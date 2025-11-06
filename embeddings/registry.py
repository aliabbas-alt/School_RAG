from __future__ import annotations
from typing import Any
from .base import EmbeddingsProvider
from .openai_embedder import OpenAIProvider, OpenAIConfig


def build_provider(name: str = "openai", **kwargs: Any) -> EmbeddingsProvider:
    """
    Simple factory so the app never imports vendor-specific code elsewhere.

    Examples:
      build_provider()  -> default openai small
      build_provider("openai", model="text-embedding-3-large", dimensions=1024)
    """
    name = (name or "openai").lower()

    if name == "openai":
        cfg = OpenAIConfig(
            model=kwargs.get("model", "text-embedding-3-small"),
            dimensions=kwargs.get("dimensions"),
            api_key_env=kwargs.get("api_key_env", "OPENAI_API_KEY"),
        )
        return OpenAIProvider(cfg)

    # Future: add "voyage", "ollama" providers:
    # if name == "voyage": return VoyageProvider(VoyageConfig(...))
    # if name == "ollama": return OllamaProvider(OllamaConfig(...))

    raise ValueError(f"Unknown embeddings provider: {name}")