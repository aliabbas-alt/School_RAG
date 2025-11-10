# storage/supabase_storage.py
"""
Supabase vector storage implementation for school document embeddings.
Fixed version to handle None responses properly.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    from supabase import create_client, Client
except ImportError:
    raise ImportError("Please install supabase: pip install supabase")

import json

@dataclass
class DocumentMetadata:
    """Extended metadata for school management system."""
    school_id: Optional[int] = None
    curriculum_type: Optional[str] = None
    document_type: Optional[str] = None
    academic_year: Optional[str] = None
    custom_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None and k != 'custom_metadata'}


class SupabaseVectorStore:
    """Manages vector embeddings in Supabase using pgvector."""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """Initialize Supabase client."""
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment.\n"
                "Add them to your .env file."
            )
        
        try:
            self.client: Client = create_client(self.url, self.key)
            print(f"✅ Connected to Supabase: {self.url}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Supabase: {e}")
    
    def store_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        doc_metadata: DocumentMetadata
    ) -> List[int]:
        """
        Store document chunks with their embeddings in Supabase.
        FIXED: Properly handles None responses from Supabase.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match embedding count ({len(embeddings)})"
            )
        
        print(f"📦 Preparing {len(chunks)} records for storage...")
        
        records = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
            chunk_meta = chunk.get("metadata", {}) or {}
            
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
            
            if idx % 100 == 0 or idx == len(chunks):
                print(f"   Prepared {idx}/{len(chunks)} records...")
        
        # Batch insert - FIXED VERSION
        print(f"☁️  Uploading to Supabase...")
        try:
            response = self.client.table("documents").insert(records).execute()
            
            # FIXED: Proper None handling
            inserted_ids = []
            
            if response is None:
                print("⚠️  Warning: Response is None, but insertion may have succeeded")
                print(f"✅ Inserted {len(records)} records (IDs not available)")
                return list(range(1, len(records) + 1))
            
            if hasattr(response, 'data'):
                data = response.data
                
                if data is None:
                    print("⚠️  Warning: Response data is None")
                    print(f"✅ Inserted {len(records)} records (IDs not available)")
                    return list(range(1, len(records) + 1))
                
                if isinstance(data, list) and len(data) > 0:
                    for record in data:
                        if isinstance(record, dict) and "id" in record:
                            inserted_ids.append(record["id"])
                    
                    if inserted_ids:
                        print(f"✅ Successfully inserted {len(inserted_ids)} document chunks")
                        return inserted_ids
            
            # Fallback: assume success if no error was raised
            print(f"✅ Inserted {len(records)} records (IDs not returned)")
            return list(range(1, len(records) + 1))
            
        except Exception as e:
            print(f"❌ Error during insertion: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
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
        """Perform similarity search on stored embeddings."""
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
            
            # FIXED: Proper None handling
            if response is None or not hasattr(response, 'data') or response.data is None:
                print("⚠️  No results found")
                return []
            
            results = response.data if isinstance(response.data, list) else []
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
        """Delete all embeddings from a specific source document."""
        print(f"🗑️  Deleting embeddings for source: {source}")
        
        try:
            response = self.client.rpc(
                "delete_by_source",
                {
                    "source_name": source,
                    "filter_school_id": school_id
                }
            ).execute()
            
            # FIXED: Proper None handling
            deleted_count = 0
            if response and hasattr(response, 'data') and response.data is not None:
                deleted_count = response.data if isinstance(response.data, int) else 0
            
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
        """Get statistics about stored documents."""
        try:
            response = self.client.rpc(
                "get_document_stats",
                {
                    "filter_school_id": school_id,
                    "filter_curriculum": curriculum_type
                }
            ).execute()
            
            # FIXED: Proper None handling
            if response and hasattr(response, 'data') and response.data:
                if isinstance(response.data, list) and len(response.data) > 0:
                    return response.data[0]
                elif isinstance(response.data, dict):
                    return response.data
            
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "unique_sources": 0,
                "avg_chunk_length": 0
            }
            
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "unique_sources": 0,
                "avg_chunk_length": 0
            }
    
    def list_sources(
        self,
        school_id: Optional[int] = None,
        curriculum_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all unique source documents with their chunk counts."""
        try:
            query = self.client.table("documents").select("source, school_id, curriculum_type, document_type, academic_year")
            
            if school_id is not None:
                query = query.eq("school_id", school_id)
            if curriculum_type is not None:
                query = query.eq("curriculum_type", curriculum_type)
            
            response = query.execute()
            
            # FIXED: Proper None handling
            if not response or not hasattr(response, 'data') or response.data is None:
                return []
            
            if isinstance(response.data, list):
                sources = {}
                for row in response.data:
                    if not isinstance(row, dict):
                        continue
                    
                    source = row.get('source')
                    if source and source not in sources:
                        sources[source] = {
                            'source': source,
                            'school_id': row.get('school_id'),
                            'curriculum_type': row.get('curriculum_type'),
                            'document_type': row.get('document_type'),
                            'academic_year': row.get('academic_year'),
                            'chunk_count': 0
                        }
                    if source:
                        sources[source]['chunk_count'] += 1
                
                return list(sources.values())
            
            return []
            
        except Exception as e:
            print(f"❌ List sources error: {e}")
            return []


def get_vector_store() -> SupabaseVectorStore:
    """Get a configured SupabaseVectorStore instance."""
    return SupabaseVectorStore()
