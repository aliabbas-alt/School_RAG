# ingest/chunk_utils.py
from __future__ import annotations
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 180,
) -> List[Document]:
    """
    Split Documents into smaller chunks tuned for RAG.

    Returns a new list of chunked Documents. Source + page metadata are preserved.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata.setdefault("source", c.metadata.get("source", "unknown"))
        if "page" in c.metadata:
            c.metadata["page"] = c.metadata["page"]
    return chunks


def summarize_chunks(chunks: List[Document]) -> Tuple[int, int, int]:
    """
    Return (num_chunks, min_len, max_len) for quick sanity checks.
    """
    lengths = [len(c.page_content) for c in chunks]
