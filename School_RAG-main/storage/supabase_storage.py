# storage/supabase_storage.py
"""
Supabase vector storage implementation for school document embeddings.
Enhanced with grade and subject tracking for educational content.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from supabase import create_client, Client
import json


@dataclass
class DocumentMetadata:
    """Extended metadata for school management system with grade and subject."""
    school_id: Optional[int] = None
    curriculum_type: Optional[str] = None  # CBSE, SSE, ICSE, IB, etc.
    document_type: Optional[str] = None  # curriculum, policy, handbook, syllabus
    academic_year: Optional[str] = None  # e.g., "2025-26"
    grade: Optional[int] = None  # Grade level: 1-12
    subject: Optional[str] = None  # English, Math, Physics, Chemistry, etc.
    custom_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None and k != 'custom_metadata'}


class SupabaseVectorStore:
    """
    Manages vector embeddings in Supabase using pgvector.
    
    Features:
    - Batch embedding storage with metadata
    - Similarity search with grade/subject filters
    - Document management (delete, stats)
    - Multi-tenant support for schools
    """
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase project URL (defaults to SUPABASE_URL env var)
            key: Supabase anon/service key (defaults to SUPABASE_KEY env var)
        
        Raises:
            EnvironmentError: If credentials are not provided
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment or passed as arguments.\n"
                "Add them to your .env file or export as environment variables."
            )
        
        self.client: Client = create_client(self.url, self.key)
        print(f"✅ Connected to Supabase: {self.url}")
    
    def store_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        doc_metadata: DocumentMetadata,
        table_name: Optional[str] = None   # NEW: allow override
    ) -> List[int]:
        """
        Store document chunks with their embeddings in Supabase.
        Routes to math_documents if subject is Math, else documents.

        Args:
            chunks: List of chunk dictionaries with 'page_content' and 'metadata'
            embeddings: List of embedding vectors (one per chunk)
            doc_metadata: School-specific metadata including grade and subject
            table_name: Optional explicit table override

        Returns:
            List of inserted document IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match embedding count ({len(embeddings)})"
            )

        print(f"📦 Preparing {len(chunks)} records for storage...")

        # --- NEW: subject-aware routing ---
        if table_name is None:
            if doc_metadata.subject and doc_metadata.subject.lower() == "math":
                target_table = "math_documents"
            else:
                target_table = "documents"
        else:
            target_table = table_name

        print(f"   📦 Target table: {target_table}")

        records = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
            chunk_meta = chunk.get("metadata", {}) or {}

            combined_metadata = {
                **chunk_meta,
                **(doc_metadata.custom_metadata or {})
            }
            for field in ['source', 'page', 'content_type', 'source_type', 'image_path']:
                combined_metadata.pop(field, None)

            record = {
                "source": chunk_meta.get("source", "unknown"),
                "page": chunk_meta.get("page"),
                "content": chunk["page_content"],
                "metadata": combined_metadata,
                "embedding": embedding,
                "school_id": doc_metadata.school_id,
                "curriculum_type": doc_metadata.curriculum_type,
                "document_type": doc_metadata.document_type,
                "academic_year": doc_metadata.academic_year,
                "grade": doc_metadata.grade,
                "subject": doc_metadata.subject,
                "content_type": chunk_meta.get("content_type", "text"),
                "source_type": chunk_meta.get("source_type", "pdf_text_extraction"),
                "image_path": chunk_meta.get("image_path")
            }
            records.append(record)

            if idx % 100 == 0 or idx == len(chunks):
                print(f"   Prepared {idx}/{len(chunks)} records...")

        print(f"☁️  Uploading to Supabase...")
        try:
            response = self.client.table(target_table).insert(records).execute()

            if hasattr(response, 'data') and response.data:
                inserted_ids = [record["id"] for record in response.data]
                print(f"✅ Successfully inserted {len(inserted_ids)} document chunks")
                return inserted_ids
            else:
                raise Exception(f"Failed to insert embeddings: {response}")
        except Exception as e:
            print(f"❌ Error during insertion: {e}")
            raise
    
    def similarity_search(
        self,
        query_embedding: List[float],
        match_threshold: float = 0.7,
        match_count: int = 10,
        school_id: Optional[int] = None,
        curriculum_type: Optional[str] = None,
        document_type: Optional[str] = None,
        academic_year: Optional[str] = None,
        grade: Optional[int] = None,   # NEW
        subject: Optional[str] = None  # NEW
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search on stored embeddings with grade/subject filters.
        Routes to match_math_documents if subject is Math, else match_documents.
        """

        # Build filter display
        filters = []
        if grade:
            filters.append(f"Grade {grade}")
        if subject:
            filters.append(subject)

        filter_str = f" [{', '.join(filters)}]" if filters else ""
        print(f"🔍 Searching{filter_str} with threshold={match_threshold}, limit={match_count}")

        try:
            # --- NEW: subject-aware RPC routing ---
            if subject and subject.lower() == "math":
                rpc_fn = "match_math_documents"
            else:
                rpc_fn = "match_documents"

            response = self.client.rpc(
                rpc_fn,
                {
                    "query_embedding": query_embedding,
                    "match_threshold": match_threshold,
                    "match_count": match_count,
                    "filter_school_id": school_id,
                    "filter_curriculum": curriculum_type,
                    "filter_document_type": document_type,
                    "filter_academic_year": academic_year,
                    "filter_grade": grade,   # NEW
                    "filter_subject": subject  # NEW
                }
            ).execute()

            results = response.data if hasattr(response, 'data') else []
            print(f"✅ Found {len(results)} matching documents from {rpc_fn}")
            return results

        except Exception as e:
            print(f"❌ Search error: {e}")
            raise
    
    def delete_by_source(
        self, 
        source: str, 
        school_id: Optional[int] = None
    ) -> int:
        """
        Delete all embeddings from a specific source document.
        
        Args:
            source: Source document name to delete
            school_id: Optional school ID filter
            
        Returns:
            Number of records deleted
        """
        print(f"🗑️  Deleting embeddings for source: {source}")
        
        try:
            response = self.client.rpc(
                "delete_by_source",
                {
                    "source_name": source,
                    "filter_school_id": school_id
                }
            ).execute()
            
            deleted_count = response.data if hasattr(response, 'data') else 0
            print(f"✅ Deleted {deleted_count} records")
            return deleted_count
        except Exception as e:
            print(f"❌ Delete error: {e}")
            raise

    def get_document_stats(
        self,
        school_id: Optional[int] = None,
        curriculum_type: Optional[str] = None,
        grade: Optional[int] = None,  # NEW
        subject: Optional[str] = None  # NEW
    ) -> Dict[str, Any]:
        """
        Get statistics about stored documents with grade/subject filters.
        
        Args:
            school_id: Optional school ID filter
            curriculum_type: Optional curriculum filter
            grade: Optional grade filter
            subject: Optional subject filter
            
        Returns:
            Dictionary with document statistics
        """
        try:
            response = self.client.rpc(
                "get_document_stats",
                {
                    "filter_school_id": school_id,
                    "filter_curriculum": curriculum_type,
                    "filter_grade": grade,  # NEW
                    "filter_subject": subject  # NEW
                }
            ).execute()
            
            if hasattr(response, 'data') and response.data:
                return response.data[0]
            return {}
        except Exception as e:
            print(f"❌ Stats error: {e}")
            raise
    
    def list_sources(
        self,
        school_id: Optional[int] = None,
        curriculum_type: Optional[str] = None,
        grade: Optional[int] = None,   # NEW
        subject: Optional[str] = None  # NEW
    ) -> List[Dict[str, Any]]:
        """
        List all unique source documents with their chunk counts.
        Routes to math_documents if subject is Math, else documents.

        Args:
            school_id: Optional school ID filter
            curriculum_type: Optional curriculum filter
            grade: Optional grade filter
            subject: Optional subject filter

        Returns:
            List of source documents with metadata
        """

        # --- NEW: subject-aware table selection ---
        target_table = "math_documents" if subject and subject.lower() == "math" else "documents"
        print(f"📄 Listing sources from table: {target_table}")

        query = self.client.table(target_table).select(
            "source, school_id, curriculum_type, document_type, academic_year, grade, subject"
        )

        if school_id is not None:
            query = query.eq("school_id", school_id)
        if curriculum_type is not None:
            query = query.eq("curriculum_type", curriculum_type)
        if grade is not None:
            query = query.eq("grade", grade)
        if subject is not None:
            query = query.eq("subject", subject)

        response = query.execute()

        if hasattr(response, 'data'):
            sources = {}
            for row in response.data:
                source = row['source']
                if source not in sources:
                    sources[source] = {
                        'source': source,
                        'school_id': row['school_id'],
                        'curriculum_type': row['curriculum_type'],
                        'document_type': row['document_type'],
                        'academic_year': row['academic_year'],
                        'grade': row.get('grade'),
                        'subject': row.get('subject'),
                        'chunk_count': 0
                    }
                sources[source]['chunk_count'] += 1

            return list(sources.values())
        return []


# Convenience function for quick initialization
def get_vector_store() -> SupabaseVectorStore:
    """Get a configured SupabaseVectorStore instance."""
    return SupabaseVectorStore()
