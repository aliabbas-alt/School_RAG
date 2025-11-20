# ai_services/openai_service.py
"""
OpenAI Service Implementation - Production Ready

Features:
- Automatic retry with exponential backoff
- Request caching for cost savings
- Rate limiting to prevent quota exhaustion
- Streaming support for real-time responses
- Comprehensive error handling
- Token tracking and cost estimation
- Detailed metrics and monitoring
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI, OpenAIError, RateLimitError, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import (
    AIServiceInterface, 
    StandardRequest, 
    StandardResponse, 
    ResponseStatus, 
    AIProvider
)
from .exceptions import (
    AIRateLimitExceeded, 
    AIServiceUnavailable, 
    AIServiceException,
    AIAuthenticationError
)
from .cache import get_cache
from .rate_limiter import RateLimiter
from .streaming import StreamingResponse

logger = logging.getLogger(__name__)


class OpenAIService(AIServiceInterface):
    """
    OpenAI service implementation with comprehensive features.
    
    Features:
    - Automatic retry with exponential backoff
    - Response caching (1-hour TTL)
    - Rate limiting (prevents quota exhaustion)
    - Streaming support
    - Cost estimation and tracking
    - Detailed metrics
    """
    
    # Pricing per 1K tokens (as of Nov 2024)
    PRICING = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "text-embedding-3-small": {"input": 0.00002, "output": 0},
        "text-embedding-3-large": {"input": 0.00013, "output": 0},
        "text-embedding-ada-002": {"input": 0.0001, "output": 0},
    }
    
    # Rate limits (requests per minute, tokens per minute)
    RATE_LIMITS = {
        "gpt-4o": (500, 30000),
        "gpt-4o-mini": (500, 200000),
        "gpt-4-turbo": (500, 30000),
        "gpt-4": (500, 10000),
        "gpt-3.5-turbo": (3500, 90000),
    }
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        default_model: str = "gpt-4o-mini",
        enable_caching: bool = True,
        enable_rate_limiting: bool = True
    ):
        """
        Initialize OpenAI service.
        
        Args:
            api_key: OpenAI API key (defaults to env var)
            default_model: Default model to use
            enable_caching: Enable response caching
            enable_rate_limiting: Enable rate limiting
        """
        # API setup
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise AIAuthenticationError("OpenAI API key not found in environment")
        
        self.client = OpenAI(api_key=self.api_key)
        self.default_model = default_model
        self.provider = AIProvider.OPENAI.value
        
        # Features
        self.enable_caching = enable_caching
        self.enable_rate_limiting = enable_rate_limiting
        
        # Rate limiter
        if enable_rate_limiting:
            rpm, tpm = self.RATE_LIMITS.get(default_model, (500, 90000))
            self.rate_limiter = RateLimiter(
                requests_per_minute=rpm,
                tokens_per_minute=tpm
            )
        else:
            self.rate_limiter = None
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cached_requests = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.total_processing_time = 0.0
        
        logger.info(f"OpenAI service initialized: model={default_model}, "
                   f"caching={enable_caching}, rate_limiting={enable_rate_limiting}")
    
    def generate_text(self, request: StandardRequest) -> StandardResponse:
        """
        Generate text completion with all features enabled.
        
        Args:
            request: Standardized request object
        
        Returns:
            StandardResponse with generated text
        """
        self.total_requests += 1
        
        # Check cache first
        if self.enable_caching:
            cache = get_cache()
            cached_response = cache.get(request)
            if cached_response:
                self.cached_requests += 1
                logger.info("✅ Returning cached response (saved API call + cost)")
                return cached_response
        
        # Rate limiting
        if self.enable_rate_limiting and self.rate_limiter:
            estimated_tokens = self._estimate_tokens(request)
            self.rate_limiter.wait_if_needed(estimated_tokens)
        
        # Make API call
        try:
            response = self._make_api_call(request)
            
            # Cache successful response
            if self.enable_caching and response.status == ResponseStatus.SUCCESS:
                cache.set(request, response)
            
            return response
        
        except Exception as e:
            self.failed_requests += 1
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError))
    )
    def _make_api_call(self, request: StandardRequest) -> StandardResponse:
        """
        Make actual OpenAI API call with retry logic.
        
        Args:
            request: Standardized request
        
        Returns:
            StandardResponse
        """
        start_time = time.time()
        
        try:
            # Build messages
            messages = self._build_messages(request)
            
            # Get parameters
            model = request.parameters.get("model", self.default_model)
            temperature = request.parameters.get("temperature", 0.2)
            max_tokens = request.parameters.get("max_tokens", 2000)
            
            logger.info(f"OpenAI API call: model={model}, temp={temperature}, "
                       f"max_tokens={max_tokens}")
            
            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract response
            content = response.choices[0].message.content
            
            # Token usage
            tokens_used = {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
            
            # Calculate cost
            cost = self.estimate_cost(tokens_used["total"], model)
            
            # Update metrics
            self.successful_requests += 1
            self.total_tokens_used += tokens_used["total"]
            self.total_cost += cost
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.info(f"✅ OpenAI success: tokens={tokens_used['total']}, "
                       f"cost=${cost:.4f}, time={processing_time:.2f}s")
            
            return StandardResponse(
                content=content,
                provider=self.provider,
                model=model,
                tokens_used=tokens_used,
                processing_time=processing_time,
                status=ResponseStatus.SUCCESS,
                cost_estimate=cost
            )
        
        except RateLimitError as e:
            logger.error(f"❌ OpenAI rate limit exceeded: {e}")
            raise AIRateLimitExceeded(f"Rate limit exceeded: {e}")
        
        except APITimeoutError as e:
            logger.error(f"❌ OpenAI timeout: {e}")
            raise AIServiceUnavailable(f"Service timeout: {e}")
        
        except APIConnectionError as e:
            logger.error(f"❌ OpenAI connection error: {e}")
            raise AIServiceUnavailable(f"Connection error: {e}")
        
        except OpenAIError as e:
            logger.error(f"❌ OpenAI API error: {e}")
            raise AIServiceException(f"OpenAI error: {e}")
        
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}", exc_info=True)
            raise AIServiceException(f"Unexpected error: {e}")
    
    def generate_text_stream(self, request: StandardRequest) -> StreamingResponse:
        """
        Generate text with streaming for real-time display.
        
        Args:
            request: Standardized request
        
        Returns:
            StreamingResponse iterator
        """
        self.total_requests += 1
        
        # Rate limiting
        if self.enable_rate_limiting and self.rate_limiter:
            estimated_tokens = self._estimate_tokens(request)
            self.rate_limiter.wait_if_needed(estimated_tokens)
        
        try:
            # Build messages
            messages = self._build_messages(request)
            
            # Get parameters
            model = request.parameters.get("model", self.default_model)
            temperature = request.parameters.get("temperature", 0.2)
            max_tokens = request.parameters.get("max_tokens", 2000)
            
            logger.info(f"OpenAI streaming: model={model}")
            
            # Make streaming API call
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True  # ← Enable streaming
            )
            
            return StreamingResponse(stream, self.provider, model)
        
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Streaming error: {e}")
            raise AIServiceException(f"Streaming error: {e}")
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        parameters: Dict[str, Any]
    ) -> StandardResponse:
        """
        Direct chat completion (alternative interface).
        
        Args:
            messages: List of message dicts with role and content
            parameters: Model parameters
        
        Returns:
            StandardResponse
        """
        # Convert to StandardRequest format
        request = StandardRequest(
            prompt=messages[-1]["content"],
            context="\n".join([m["content"] for m in messages[:-1] if m["role"] == "system"]),
            parameters=parameters
        )
        
        return self.generate_text(request)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embeddings(
        self, 
        text: str, 
        model: str = "text-embedding-3-small"
    ) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            model: Embedding model to use
        
        Returns:
            List of embedding values
        """
        try:
            logger.info(f"Generating embeddings with {model}")
            
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            
            embeddings = response.data[0].embedding
            tokens_used = response.usage.total_tokens
            
            # Update metrics
            self.total_tokens_used += tokens_used
            cost = self.estimate_cost(tokens_used, model)
            self.total_cost += cost
            
            logger.info(f"✅ Embeddings generated: tokens={tokens_used}, "
                       f"cost=${cost:.6f}")
            
            return embeddings
        
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise AIServiceException(f"Embedding error: {e}")
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """
        Estimate cost for token usage.
        
        Args:
            tokens: Number of tokens
            model: Model name
        
        Returns:
            Estimated cost in USD
        """
        if model not in self.PRICING:
            logger.warning(f"Unknown model {model}, using default pricing")
            return (tokens / 1000) * 0.001  # Default conservative estimate
        
        pricing = self.PRICING[model]
        
        # Simplified: assume 50/50 split for input/output
        # More accurate tracking would require separate input/output counts
        cost = (tokens / 1000) * ((pricing["input"] + pricing["output"]) / 2)
        
        return cost
    
    def _build_messages(self, request: StandardRequest) -> List[Dict[str, str]]:
        """Build messages array from request."""
        messages = []
        
        # Add system context if provided
        if request.context:
            messages.append({
                "role": "system",
                "content": request.context
            })
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": request.prompt
        })
        
        return messages
    
    def _estimate_tokens(self, request: StandardRequest) -> int:
        """
        Estimate token count for request.
        Uses rough approximation: 1 token ≈ 4 characters.
        
        Args:
            request: StandardRequest
        
        Returns:
            Estimated token count
        """
        total_chars = len(request.prompt)
        if request.context:
            total_chars += len(request.context)
        
        # Add estimated completion tokens
        max_tokens = request.parameters.get("max_tokens", 2000)
        
        estimated_tokens = (total_chars // 4) + max_tokens
        
        return estimated_tokens
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive service metrics.
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "provider": self.provider,
            "default_model": self.default_model,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cached_requests": self.cached_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "cache_hit_rate": self.cached_requests / max(self.total_requests, 1),
            "total_tokens_used": self.total_tokens_used,
            "total_cost": round(self.total_cost, 4),
            "average_tokens_per_request": self.total_tokens_used / max(self.successful_requests, 1),
            "average_cost_per_request": self.total_cost / max(self.successful_requests, 1),
            "average_processing_time": self.total_processing_time / max(self.successful_requests, 1),
            "features": {
                "caching": self.enable_caching,
                "rate_limiting": self.enable_rate_limiting
            }
        }
        
        # Add cache metrics
        if self.enable_caching:
            cache = get_cache()
            metrics["cache_metrics"] = cache.get_metrics()
        
        # Add rate limiter metrics
        if self.enable_rate_limiting and self.rate_limiter:
            metrics["rate_limiter_metrics"] = self.rate_limiter.get_metrics()
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics counters."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cached_requests = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.total_processing_time = 0.0
        
        logger.info("Metrics reset")
