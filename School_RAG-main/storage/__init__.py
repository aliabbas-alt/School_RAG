# storage/__init__.py
"""
Storage package for persisting embeddings to various backends.
"""

from .supabase_storage import SupabaseVectorStore, DocumentMetadata, get_vector_store

__all__ = ['SupabaseVectorStore', 'DocumentMetadata', 'get_vector_store']
