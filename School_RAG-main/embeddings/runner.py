from __future__ import annotations
import os, json
from typing import List, Dict, Any
import pandas as pd
from .registry import build_provider
from .base import ChunkRecord


def load_chunks(json_path: str) -> List[ChunkRecord]:
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
    base, _ = os.path.splitext(json_path)
    safe_model = model.replace("-", "")
    return f"{base}__{provider}_{safe_model}{ext}"


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
    chunks = load_chunks(json_path)
    texts = [c.page_content for c in chunks]

    provider = build_provider(provider_name, model=model, dimensions=dimensions)
    result = provider.embed_texts(texts)

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

    ext = os.path.splitext(out_path)[1].lower()
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

