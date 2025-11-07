from __future__ import annotations
import os
from typing import List
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from .base import EmbeddingsProvider, EmbeddingResult


SUPPORTED_OPENAI_MODELS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


@dataclass
class OpenAIConfig:
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    api_key_env: str = "OPENAI_API_KEY"


class OpenAIProvider(EmbeddingsProvider):
    def __init__(self, cfg: OpenAIConfig = OpenAIConfig()):
        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"{cfg.api_key_env} not set. Add it to your environment or .env file."
            )
        if cfg.model not in SUPPORTED_OPENAI_MODELS:
            raise ValueError(f"Unsupported OpenAI embedding model: {cfg.model}")

        default_dim = SUPPORTED_OPENAI_MODELS[cfg.model]
        if cfg.dimensions is not None:
            if cfg.dimensions <= 0:
                raise ValueError("Dimensions override must be a positive integer.")
            if cfg.dimensions > default_dim:
                # OpenAI v3 embeddings support shortening to <= default dimension.
                raise ValueError(
                    f"Dimensions override ({cfg.dimensions}) must be ≤ model's default ({default_dim})."
                )

        extra_args = {}
        if cfg.dimensions is not None:
            extra_args["dimensions"] = cfg.dimensions  # OpenAI supports this

        self._model_id = cfg.model
        self._dimension = cfg.dimensions or default_dim
        self._emb = OpenAIEmbeddings(model=cfg.model, **extra_args)

    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        vectors = self._emb.embed_documents(texts)
        return EmbeddingResult(
            vectors=vectors,
            model=self._model_id,
            dimension=self._dimension,
            provider="openai",
        )
