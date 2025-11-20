# ai_services/streaming.py
"""Streaming support for real-time responses."""

from typing import Iterator, Dict, Any, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """A single chunk in a streaming response."""
    content: str
    delta: str  # Just the new content
    tokens_so_far: int
    finish_reason: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class StreamingResponse:
    """Wrapper for streaming responses with metrics."""
    
    def __init__(self, stream: Iterator, provider: str, model: str):
        self.stream = stream
        self.provider = provider
        self.model = model
        self.start_time = time.time()
        self.total_tokens = 0
        self.full_content = ""
    
    def __iter__(self) -> Iterator[StreamChunk]:
        """Stream chunks with automatic metrics tracking."""
        try:
            for chunk in self.stream:
                # Provider-specific extraction (OpenAI format)
                delta = chunk.choices[0].delta.content or ""
                self.full_content += delta
                self.total_tokens += 1  # Approximate
                
                yield StreamChunk(
                    content=self.full_content,
                    delta=delta,
                    tokens_so_far=self.total_tokens,
                    finish_reason=chunk.choices[0].finish_reason
                )
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming session metrics."""
        return {
            "provider": self.provider,
            "model": self.model,
            "total_tokens": self.total_tokens,
            "duration": time.time() - self.start_time,
            "content_length": len(self.full_content)
        }
