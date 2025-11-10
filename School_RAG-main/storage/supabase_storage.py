# storage/supabase_storage.py
"""
Supabase vector storage implementation for school document embeddings.
Integrates with existing document processing pipeline.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from supabase import create_client, Client
import json

@dataclass
class DocumentMetadata:
    """Extended metadata for school management system."""
    school_id: Optional[int] = None
    curriculum_type: Optional[str] = None  # CBSE, SSE, ICSE, IB, etc.
    document_type: Optional[str] = None  # curriculum, policy, handbook, syllabus
    academic_year: Optional[str] = None  # e.g., "2025-26"
    custom_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None and k != 'custom_metadata'}


class SupabaseVectorStore:
    """
    Manages vector embeddings in Supabase using pgvector.
    
    Features:
    - Batch embedding storage with metadata
    - Similarity search with multiple filters
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
        doc_metadata: DocumentMetadata
    ) -> List[int]:
        """
        Store document chunks with their embeddings in Supabase.
        
        Args:
            chunks: List of chunk dictionaries with 'page_content' and 'metadata'
            embeddings: List of embedding vectors (one per chunk)
            doc_metadata: School-specific metadata for all chunks
            
        Returns:
            List of inserted document IDs
            
        Raises:
            ValueError: If chunk count doesn't match embedding count
            Exception: If insertion fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match embedding count ({len(embeddings)})"
            )
        
        print(f"📦 Preparing {len(chunks)} records for storage...")
        
        records = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
            # Extract chunk metadata
            chunk_meta = chunk.get("metadata", {}) or {}
            
            # Combine all metadata
            combined_metadata = {
                **chunk_meta,
                **(doc_metadata.custom_metadata or {})
            }
            
            # Remove fields that have dedicated columns
            for field in ['source', 'page']:
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
                "academic_year": doc_metadata.academic_year
            }
            records.append(record)
            
            # Progress indicator
            if idx % 100 == 0 or idx == len(chunks):
                print(f"   Prepared {idx}/{len(chunks)} records...")
        
        # Batch insert all records
        print(f"☁️  Uploading to Supabase...")
        try:
            response = self.client.table("documents").insert(records).execute()
            
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
        academic_year: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search on stored embeddings.
        
        Args:
            query_embedding: Query vector to search for
            match_threshold: Minimum similarity score (0-1)
            match_count: Maximum number of results
            school_id: Filter by school (None = all schools)
            curriculum_type: Filter by curriculum (None = all)
            document_type: Filter by document type (None = all)
            academic_year: Filter by academic year (None = all)
            
        Returns:
            List of matching documents with similarity scores
        """
        print(f"🔍 Searching with threshold={match_threshold}, limit={match_count}")
        
        try:
            response = self.client.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": match_threshold,
                    "match_count": match_count,
                    "filter_school_id": school_id,
                    "filter_curriculum": curriculum_type,
                    "filter_document_type": document_type,
                    "filter_academic_year": academic_year
                }
            ).execute()
            
            results = response.data if hasattr(response, 'data') else []
            print(f"✅ Found {len(results)} matching documents")
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
        curriculum_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about stored documents.
        
        Args:
            school_id: Optional school ID filter
            curriculum_type: Optional curriculum filter
            
        Returns:
            Dictionary with document statistics
        """
        try:
            response = self.client.rpc(
                "get_document_stats",
                {
                    "filter_school_id": school_id,
                    "filter_curriculum": curriculum_type
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
        curriculum_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all unique source documents with their chunk counts.
        
        Args:
            school_id: Optional school ID filter
            curriculum_type: Optional curriculum filter
            
        Returns:
            List of source documents with metadata
        """
        query = self.client.table("documents").select("source, school_id, curriculum_type, document_type, academic_year")
        
        if school_id is not None:
            query = query.eq("school_id", school_id)
        if curriculum_type is not None:
            query = query.eq("curriculum_type", curriculum_type)
        
        response = query.execute()
        
        if hasattr(response, 'data'):
            # Group by source and count
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
                        'chunk_count': 0
                    }
                sources[source]['chunk_count'] += 1
            
            return list(sources.values())
        return []


# Convenience function for quick initialization
def get_vector_store() -> SupabaseVectorStore:
    """Get a configured SupabaseVectorStore instance."""
    return SupabaseVectorStore()
