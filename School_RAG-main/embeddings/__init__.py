# embeddings/__init__.py
"""
Embeddings package for generating vector embeddings from text.
"""

from .base import EmbeddingsProvider, EmbeddingResult, ChunkRecord
from .registry import build_provider
from .runner import load_chunks, embed_and_save, embed_and_store_supabase

__all__ = [
    'EmbeddingsProvider',
    'EmbeddingResult',
    'ChunkRecord',
    'build_provider',
    'load_chunks',
    'embed_and_save',
    'embed_and_store_supabase'
]
