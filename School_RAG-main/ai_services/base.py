"""Base classes and interfaces for AI services."""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


class AIProvider(Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    LOCAL = "local"


class ResponseStatus(Enum):
    """Response status indicators."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


@dataclass
class StandardRequest:
    """Standardized request format across all AI providers."""
    prompt: str
    context: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.2,
        "max_tokens": 2000
    })
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "school_id": None,
        "user_id": None,
        "curriculum_type": None,
        "request_type": "general"
    })


@dataclass
class StandardResponse:
    """Standardized response format across all AI providers."""
    content: str
    provider: str
    model: str
    tokens_used: Dict[str, int]
    processing_time: float
    status: ResponseStatus
    cost_estimate: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AIServiceInterface(ABC):
    """Abstract base class for all AI service implementations."""
    
    @abstractmethod
    def generate_text(self, request: StandardRequest) -> StandardResponse:
        """Generate text completion."""
        pass
    
    @abstractmethod
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], 
                        parameters: Dict[str, Any]) -> StandardResponse:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage."""
        pass
