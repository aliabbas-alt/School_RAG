# embeddings/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Protocol


@dataclass
class ChunkRecord:
    page_content: str
    metadata: Dict[str, Any]


@dataclass
class EmbeddingResult:
    vectors: List[List[float]]   # one vector per input text
    model: str                   # e.g., "text-embedding-3-small"
    dimension: int               # e.g., 1536
    provider: str                # e.g., "openai"


class EmbeddingsProvider(Protocol):
    """Minimal provider-agnostic interface."""
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        ...