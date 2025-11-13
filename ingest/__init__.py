# ingest/__init__.py
"""
Document ingestion package for loading and processing PDFs.
"""

from .pdf_loader import load_pdf
from .chunk_utils import chunk_documents, summarize_chunks

__all__ = ['load_pdf', 'chunk_documents', 'summarize_chunks']
