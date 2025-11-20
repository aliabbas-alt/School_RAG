# embeddings/runner.py
"""
Embedding pipeline runner with Supabase integration.
Enhanced with grade and subject metadata support.
"""

from __future__ import annotations

import os
import json
import csv
from typing import List, Dict, Any, Optional
import pandas as pd
import tiktoken

from .registry import build_provider
from .base import ChunkRecord


def load_chunks(json_path: str) -> List[ChunkRecord]:
    """Load chunks from JSON file produced by run_parse.py."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    chunks = payload.get("chunks", [])
    out: List[ChunkRecord] = []
    
    for c in chunks:
        out.append(ChunkRecord(
            page_content=c.get("page_content", ""),
            metadata=c.get("metadata", {}) or {}
        ))
    
    return out


def default_embeddings_output(json_path: str, provider: str, model: str, ext: str = ".parquet") -> str:
    """Generate default output filename for embeddings."""
    base, _ = os.path.splitext(json_path)
    safe_model = model.replace("-", "")
    return f"{base}__{provider}_{safe_model}{ext}"

def count_tokens(texts: List[str], model: str) -> int:
    """
    Count total tokens for a list of texts using tiktoken.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # fallback to a default encoding if model not recognized
        enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(t)) for t in texts)

def embed_and_save(
    json_path: str,
    provider_name: str,
    model: str,
    dimensions: int | None,
    out_ext: str,
    out_path: str | None = None,
) -> str:
    """
    Load chunk JSON, compute embeddings via selected provider, and save to file.
    Returns the absolute path to the saved embeddings file.
    """
    print(f"📄 Loading chunks from {json_path}...")
    chunks = load_chunks(json_path)
    print(f"   Loaded {len(chunks)} chunks")
    
    print(f"🔮 Computing embeddings with {provider_name}/{model}...")
    texts = [c.page_content for c in chunks]
    provider = build_provider(provider_name, model=model, dimensions=dimensions)
    result = provider.embed_texts(texts)
    print(f"   Generated {len(result.vectors)} embeddings (dim={result.dimension})")
    
    # Build rows with all metadata
    rows: List[Dict[str, Any]] = []
    for c, vec in zip(chunks, result.vectors):
        rows.append({
            "text": c.page_content,
            "embedding": vec,
            "source": c.metadata.get("source"),
            "page": c.metadata.get("page"),
            "metadata": c.metadata,
            "provider": result.provider,
            "model": result.model,
            "dimension": result.dimension,
        })
    
    df = pd.DataFrame(rows)
    
    # Default output path next to the chunk JSON
    if out_path is None:
        out_path = default_embeddings_output(json_path, provider_name, model, out_ext)
    
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Save in requested format
    ext = os.path.splitext(out_path)[1].lower()
    
    print(f"💾 Saving embeddings to {out_path}...")
    if ext in [".parquet", ".pq"]:
        df.to_parquet(out_path, index=False)
    elif ext in [".csv"]:
        df.to_csv(out_path, index=False)
    elif ext in [".json", ".ndjson"]:
        if ext == ".ndjson":
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in rows:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
    else:
        # Fallback to parquet
        df.to_parquet(out_path + ".parquet", index=False)
        out_path = out_path + ".parquet"
    
    print(f"✅ Saved successfully!")
    return os.path.abspath(out_path)

def save_embeddings_csv(json_path: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]], result, school_id, curriculum_type, document_type, academic_year, grade, subject):
    """
    Save embeddings and metadata to a CSV file for backup.
    """
    backup_path = os.path.splitext(json_path)[0] + "_embeddings_backup.csv"
    print(f"💾 Saving CSV backup to {backup_path}...")

    with open(backup_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow([
            "chunk_index", "text", "embedding", "source", "page",
            "school_id", "curriculum_type", "document_type", "academic_year",
            "grade", "subject", "provider", "model", "dimension"
        ])

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings), 1):
            writer.writerow([
                idx,
                chunk["page_content"],
                json.dumps(emb),  # store embedding vector as JSON string
                chunk["metadata"].get("source"),
                chunk["metadata"].get("page"),
                school_id,
                curriculum_type,
                document_type,
                academic_year,
                grade,
                subject,
                result.provider,
                result.model,
                result.dimension
            ])

    print("✅ CSV backup saved successfully!")
    return backup_path

# ============================================================================
# SUPABASE INTEGRATION (Enhanced with grade and subject)
# ============================================================================


def embed_and_store_supabase(
    json_path: str,
    school_id: int,
    curriculum_type: str,
    document_type: str = "curriculum",
    academic_year: str = "2025-26",
    grade: Optional[int] = None,  # NEW
    subject: Optional[str] = None,  # NEW
    provider_name: str = "openai",
    model: Optional[str] = None,   # allow auto-selection
    dimensions: Optional[int] = None,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete pipeline: Load chunks → Generate embeddings → Store in Supabase.
    Enhanced with grade and subject metadata.
    """

    # Import Supabase storage (only when needed)
    try:
        from storage.supabase_storage import SupabaseVectorStore, DocumentMetadata
    except ImportError:
        raise ImportError(
            "Supabase storage not available. Make sure to:\n"
            "1. Install: pip install supabase python-dotenv\n"
            "2. Create storage/supabase_storage.py with SupabaseVectorStore class"
        )

    print(f"\n{'='*60}")
    print(f"SUPABASE STORAGE PIPELINE")
    print(f"{'='*60}")

    # Step 1: Load chunks
    print(f"\n📄 Step 1: Loading chunks from {os.path.basename(json_path)}")
    chunks = load_chunks(json_path)
    print(f"   ✅ Loaded {len(chunks)} chunks")

    if len(chunks) == 0:
        raise ValueError("No chunks found in JSON file")

    # --- NEW: subject-aware model selection ---
    if model is None:
        if subject and subject.lower() == "math":
            model = "text-embedding-3-large"
        else:
            model = "text-embedding-3-small"

    # Step 2: Generate embeddings
    print(f"\n🔮 Step 2: Computing embeddings")
    print(f"   Provider: {provider_name}")
    print(f"   Model: {model}")
    print(f"   Dimensions: {dimensions or 'default'}")

    provider = build_provider(provider_name, model=model, dimensions=dimensions)
    texts = [c.page_content for c in chunks]

    try:
        token_count = count_tokens(texts, model)
        print(f"🧮 Estimated tokens: {token_count}")
    except Exception as e:
        print(f"⚠️ Token counting failed: {e}")
        token_count = None  # fallback so workflow continues

    result = provider.embed_texts(texts)

    print(f"   ✅ Generated {len(result.vectors)} embeddings (dim={result.dimension})")

    # Step 3: Prepare chunks as dictionaries
    print(f"\n📦 Step 3: Preparing data for storage")
    chunk_dicts = [
        {
            "page_content": c.page_content,
            "metadata": c.metadata
        }
        for c in chunks
    ]

    # NEW: Save CSV backup locally
    save_embeddings_csv(
        json_path=json_path,
        chunks=chunk_dicts,
        embeddings=result.vectors,
        result=result,
        school_id=school_id,
        curriculum_type=curriculum_type,
        document_type=document_type,
        academic_year=academic_year,
        grade=grade,
        subject=subject
    )

    # Step 4: Store in Supabase with grade/subject metadata
    print(f"\n☁️  Step 4: Storing in Supabase")
    print(f"   School ID: {school_id}")
    print(f"   Curriculum: {curriculum_type}")
    print(f"   Document Type: {document_type}")
    print(f"   Academic Year: {academic_year}")

    if grade:
        print(f"   Grade: {grade}")
    if subject:
        print(f"   Subject: {subject}")

    vector_store = SupabaseVectorStore()

    doc_metadata = DocumentMetadata(
        school_id=school_id,
        curriculum_type=curriculum_type,
        document_type=document_type,
        academic_year=academic_year,
        grade=grade,
        subject=subject,
        custom_metadata=custom_metadata
    )

    inserted_ids = vector_store.store_embeddings(
        chunks=chunk_dicts,
        embeddings=result.vectors,
        doc_metadata=doc_metadata
    )

    print(f"\n{'='*60}")
    print(f"✅ PIPELINE COMPLETE")
    print(f"{'='*60}")

    summary = {
        "success": True,
        "source_file": json_path,
        "inserted_count": len(inserted_ids),
        "inserted_ids": inserted_ids,
        "embedding_model": result.model,
        "embedding_dimension": result.dimension,
        "embedding_provider": result.provider,
        "school_id": school_id,
        "curriculum_type": curriculum_type,
        "document_type": document_type,
        "academic_year": academic_year,
        "grade": grade,
        "subject": subject,
        "token_count": token_count
    }

    return summary
