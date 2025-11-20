# ai_services/manager.py
"""
AI Service Manager - Centralized AI Provider Management

Features:
- Multi-provider support (OpenAI, Gemini, Claude, etc.)
- Automatic fallback on failures
- School quota management
- Request routing (by cost, priority, etc.)
- Comprehensive metrics aggregation
- Global convenience functions
"""

import logging
from typing import Dict, Optional, List, Any
from enum import Enum

from .base import (
    AIServiceInterface, 
    AIProvider, 
    StandardRequest, 
    StandardResponse,
    ResponseStatus
)
from .openai_service import OpenAIService
from .exceptions import (
    AIServiceUnavailable, 
    AIRateLimitExceeded,
    AIQuotaExceeded,
    AIServiceException
)
from .quota_manager import get_quota_manager

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Strategy for routing requests to providers."""
    PRIMARY_ONLY = "primary_only"  # Use only primary provider
    CHEAPEST_FIRST = "cheapest_first"  # Route to cheapest provider first
    FASTEST_FIRST = "fastest_first"  # Route to fastest provider first
    ROUND_ROBIN = "round_robin"  # Distribute evenly
    FALLBACK_CHAIN = "fallback_chain"  # Try providers in order until success


class AIServiceManager:
    """
    Centralized manager for AI services.
    
    Features:
    - Multi-provider support with automatic fallback
    - School quota enforcement
    - Request routing strategies
    - Comprehensive metrics
    
    Phase 1: Single provider (OpenAI)
    Future: Multi-provider with intelligent routing
    """
    
    def __init__(
        self, 
        default_provider: AIProvider = AIProvider.OPENAI,
        fallback_chain: Optional[List[AIProvider]] = None,
        routing_strategy: RoutingStrategy = RoutingStrategy.FALLBACK_CHAIN,
        enable_quota_management: bool = False
    ):
        """
        Initialize service manager.
        
        Args:
            default_provider: Default AI provider to use
            fallback_chain: Ordered list of providers for fallback
            routing_strategy: How to route requests
            enable_quota_management: Enable per-school quota limits
        """
        self.default_provider = default_provider
        self.fallback_chain = fallback_chain or [AIProvider.OPENAI]
        self.routing_strategy = routing_strategy
        self.enable_quota_management = enable_quota_management
        
        # Initialize services
        self.services: Dict[AIProvider, AIServiceInterface] = {}
        self._initialize_services()
        
        # Round-robin counter
        self._round_robin_index = 0
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.fallback_count = 0
        self.quota_rejections = 0
        
        logger.info(f"AI Service Manager initialized: provider={default_provider.value}, "
                   f"strategy={routing_strategy.value}, quota_mgmt={enable_quota_management}")
    
    def _initialize_services(self):
        """Initialize all providers in the fallback chain."""
        for provider in self.fallback_chain:
            try:
                if provider == AIProvider.OPENAI:
                    self.services[provider] = OpenAIService()
                    logger.info(f"✅ {provider.value} service initialized")
                # Future: Add other providers
                # elif provider == AIProvider.GEMINI:
                #     self.services[provider] = GeminiService()
                # elif provider == AIProvider.CLAUDE:
                #     self.services[provider] = ClaudeService()
            except Exception as e:
                logger.error(f"❌ Failed to initialize {provider.value}: {e}")
    
    def generate_text(
        self, 
        request: StandardRequest, 
        provider: Optional[AIProvider] = None
    ) -> StandardResponse:
        """
        Generate text using specified provider with fallback support.
        
        Args:
            request: StandardRequest object
            provider: Specific provider to use (None = use routing strategy)
        
        Returns:
            StandardResponse
        
        Raises:
            AIQuotaExceeded: If school quota exceeded
            AIServiceUnavailable: If all providers fail
        """
        self.total_requests += 1
        
        # Check school quota first (if enabled)
        if self.enable_quota_management:
            self._check_school_quota(request)
        
        # Determine which provider(s) to try
        if provider:
            # Specific provider requested
            providers_to_try = [provider]
        else:
            # Use routing strategy
            providers_to_try = self._get_providers_by_strategy(request)
        
        # Try providers in order
        errors = []
        for idx, current_provider in enumerate(providers_to_try):
            if current_provider not in self.services:
                logger.warning(f"Provider {current_provider.value} not available")
                continue
            
            try:
                logger.info(f"Attempting provider: {current_provider.value} "
                           f"({idx + 1}/{len(providers_to_try)})")
                
                service = self.services[current_provider]
                response = service.generate_text(request)
                
                if response.status == ResponseStatus.SUCCESS:
                    # Success!
                    self.successful_requests += 1
                    
                    # Record usage for quota tracking
                    if self.enable_quota_management:
                        self._record_quota_usage(request, response)
                    
                    # Track fallback
                    if idx > 0:
                        self.fallback_count += 1
                        logger.info(f"✅ Fallback successful to {current_provider.value}")
                    
                    return response
            
            except AIRateLimitExceeded as e:
                logger.warning(f"⏸️  {current_provider.value} rate limited: {e}")
                errors.append(f"{current_provider.value}: Rate limit")
                continue  # Try next provider
            
            except AIServiceUnavailable as e:
                logger.warning(f"❌ {current_provider.value} unavailable: {e}")
                errors.append(f"{current_provider.value}: Unavailable")
                continue  # Try next provider
            
            except AIServiceException as e:
                logger.error(f"❌ {current_provider.value} error: {e}")
                errors.append(f"{current_provider.value}: {str(e)}")
                continue  # Try next provider
            
            except Exception as e:
                logger.error(f"❌ Unexpected error with {current_provider.value}: {e}", 
                           exc_info=True)
                errors.append(f"{current_provider.value}: Unexpected error")
                continue
        
        # All providers failed
        self.failed_requests += 1
        error_msg = " | ".join(errors) if errors else "No providers available"
        
        logger.error(f"❌ All providers failed: {error_msg}")
        raise AIServiceUnavailable(f"All providers failed: {error_msg}")
    
    def generate_embeddings(
        self, 
        text: str, 
        provider: Optional[AIProvider] = None
    ) -> List[float]:
        """
        Generate embeddings using specified provider.
        
        Args:
            text: Text to embed
            provider: Provider to use (defaults to default_provider)
        
        Returns:
            List of embedding values
        """
        provider = provider or self.default_provider
        
        if provider not in self.services:
            raise AIServiceUnavailable(f"Provider {provider.value} not available")
        
        try:
            return self.services[provider].generate_embeddings(text)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def _get_providers_by_strategy(
        self, 
        request: StandardRequest
    ) -> List[AIProvider]:
        """
        Get ordered list of providers based on routing strategy.
        
        Args:
            request: StandardRequest (may contain routing hints)
        
        Returns:
            Ordered list of providers to try
        """
        if self.routing_strategy == RoutingStrategy.PRIMARY_ONLY:
            # Only use default provider
            return [self.default_provider]
        
        elif self.routing_strategy == RoutingStrategy.FALLBACK_CHAIN:
            # Try all providers in fallback chain order
            return self.fallback_chain.copy()
        
        elif self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            # Rotate through providers
            providers = self.fallback_chain.copy()
            self._round_robin_index = (self._round_robin_index + 1) % len(providers)
            
            # Reorder to start with current index
            rotated = providers[self._round_robin_index:] + providers[:self._round_robin_index]
            return rotated
        
        elif self.routing_strategy == RoutingStrategy.CHEAPEST_FIRST:
            # Sort by estimated cost (OpenAI usually more expensive)
            # For now, simple ordering (will improve with actual cost data)
            # Gemini < OpenAI < Claude (rough estimate)
            priority = [AIProvider.OPENAI]  # Phase 1: Only OpenAI
            return [p for p in priority if p in self.services]
        
        elif self.routing_strategy == RoutingStrategy.FASTEST_FIRST:
            # Sort by historical speed (will track in future)
            # For now, use fallback chain
            return self.fallback_chain.copy()
        
        else:
            # Default to fallback chain
            return self.fallback_chain.copy()
    
    def _check_school_quota(self, request: StandardRequest):
        """
        Check if school has quota available.
        
        Args:
            request: StandardRequest with school_id in metadata
        
        Raises:
            AIQuotaExceeded: If quota exceeded
        """
        school_id = request.metadata.get("school_id")
        if not school_id:
            # No school_id = no quota check
            return
        
        quota_mgr = get_quota_manager()
        
        # Estimate tokens and cost
        estimated_tokens = len(request.prompt) / 4  # Rough estimate
        if request.context:
            estimated_tokens += len(request.context) / 4
        
        estimated_cost = estimated_tokens * 0.0001  # Very rough estimate
        
        # Check quota
        allowed, reason = quota_mgr.check_quota(
            school_id, 
            int(estimated_tokens), 
            estimated_cost
        )
        
        if not allowed:
            self.quota_rejections += 1
            logger.warning(f"❌ Quota exceeded for {school_id}: {reason}")
            raise AIQuotaExceeded(f"School quota exceeded: {reason}")
    
    def _record_quota_usage(
        self, 
        request: StandardRequest, 
        response: StandardResponse
    ):
        """Record usage for quota tracking."""
        school_id = request.metadata.get("school_id")
        if not school_id:
            return
        
        quota_mgr = get_quota_manager()
        quota_mgr.record_usage(
            school_id,
            response.tokens_used["total"],
            response.cost_estimate
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics from all services.
        
        Returns:
            Dictionary with metrics from all providers
        """
        manager_metrics = {
            "manager": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": self.successful_requests / max(self.total_requests, 1),
                "fallback_count": self.fallback_count,
                "quota_rejections": self.quota_rejections,
                "routing_strategy": self.routing_strategy.value,
                "quota_management_enabled": self.enable_quota_management
            }
        }
        
        # Add metrics from each provider
        provider_metrics = {
            provider.value: service.get_metrics()
            for provider, service in self.services.items()
        }
        
        return {**manager_metrics, **provider_metrics}
    
    def get_school_usage(self, school_id: str) -> Dict[str, Any]:
        """
        Get usage report for a specific school.
        
        Args:
            school_id: School identifier
        
        Returns:
            Usage report dictionary
        """
        if not self.enable_quota_management:
            return {"status": "Quota management not enabled"}
        
        quota_mgr = get_quota_manager()
        return quota_mgr.get_usage_report(school_id)
    
    def set_school_quota(
        self, 
        school_id: str, 
        monthly_requests: int = 10000,
        monthly_tokens: int = 1000000,
        monthly_cost: float = 100.0
    ):
        """
        Set quota for a school.
        
        Args:
            school_id: School identifier
            monthly_requests: Max requests per month
            monthly_tokens: Max tokens per month
            monthly_cost: Max cost per month (USD)
        """
        if not self.enable_quota_management:
            logger.warning("Quota management not enabled")
            return
        
        quota_mgr = get_quota_manager()
        quota_mgr.set_quota(school_id, monthly_requests, monthly_tokens, monthly_cost)
        
        logger.info(f"Quota set for {school_id}: {monthly_requests} req/month, "
                   f"${monthly_cost}/month")
    
    def reset_metrics(self):
        """Reset all metrics counters."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.fallback_count = 0
        self.quota_rejections = 0
        
        # Reset provider metrics
        for service in self.services.values():
            if hasattr(service, 'reset_metrics'):
                service.reset_metrics()
        
        logger.info("All metrics reset")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"AIServiceManager(providers={[p.value for p in self.services.keys()]}, "
                f"strategy={self.routing_strategy.value})")


# ============================================================================
# GLOBAL INSTANCE & CONVENIENCE FUNCTIONS
# ============================================================================

# Global service manager instance (singleton)
_service_manager: Optional[AIServiceManager] = None


def get_ai_service() -> AIServiceManager:
    """
    Get or create global AI service manager.
    
    Returns:
        Global AIServiceManager instance
    """
    global _service_manager
    if _service_manager is None:
        _service_manager = AIServiceManager()
    return _service_manager


def initialize_ai_service(
    default_provider: AIProvider = AIProvider.OPENAI,
    fallback_chain: Optional[List[AIProvider]] = None,
    routing_strategy: RoutingStrategy = RoutingStrategy.FALLBACK_CHAIN,
    enable_quota_management: bool = False
) -> AIServiceManager:
    """
    Initialize global AI service manager with custom configuration.
    
    Args:
        default_provider: Default provider
        fallback_chain: Fallback providers
        routing_strategy: Routing strategy
        enable_quota_management: Enable quota limits
    
    Returns:
        Configured AIServiceManager
    """
    global _service_manager
    _service_manager = AIServiceManager(
        default_provider=default_provider,
        fallback_chain=fallback_chain,
        routing_strategy=routing_strategy,
        enable_quota_management=enable_quota_management
    )
    return _service_manager


def generate_text(
    prompt: str, 
    context: Optional[str] = None,
    school_id: Optional[str] = None,
    **parameters
) -> StandardResponse:
    """
    Convenience function for text generation.
    
    Args:
        prompt: The prompt/question
        context: Optional context (system message)
        school_id: Optional school ID for quota tracking
        **parameters: Additional parameters (temperature, max_tokens, etc.)
    
    Returns:
        StandardResponse
    
    Example:
        >>> response = generate_text(
        ...     prompt="Explain photosynthesis",
        ...     context="You are a science teacher",
        ...     temperature=0.3,
        ...     school_id="school_123"
        ... )
        >>> print(response.content)
    """
    # Build metadata
    metadata = parameters.pop("metadata", {})
    if school_id:
        metadata["school_id"] = school_id
    
    # Create request
    request = StandardRequest(
        prompt=prompt,
        context=context,
        parameters=parameters,
        metadata=metadata
    )
    
    return get_ai_service().generate_text(request)


def generate_embeddings(text: str) -> List[float]:
    """
    Convenience function for embeddings generation.
    
    Args:
        text: Text to embed
    
    Returns:
        List of embedding values
    
    Example:
        >>> embeddings = generate_embeddings("Hello world")
        >>> print(len(embeddings))
        1536
    """
    return get_ai_service().generate_embeddings(text)


def get_metrics() -> Dict[str, Any]:
    """
    Get comprehensive metrics from all services.
    
    Returns:
        Dictionary with all metrics
    
    Example:
        >>> metrics = get_metrics()
        >>> print(metrics['manager']['success_rate'])
        0.98
    """
    return get_ai_service().get_metrics()


def get_school_usage(school_id: str) -> Dict[str, Any]:
    """
    Get usage report for a school.
    
    Args:
        school_id: School identifier
    
    Returns:
        Usage report
    """
    return get_ai_service().get_school_usage(school_id)


def set_school_quota(
    school_id: str,
    monthly_requests: int = 10000,
    monthly_tokens: int = 1000000,
    monthly_cost: float = 100.0
):
    """
    Set quota for a school.
    
    Args:
        school_id: School identifier
        monthly_requests: Max requests per month
        monthly_tokens: Max tokens per month
        monthly_cost: Max cost per month (USD)
    """
    get_ai_service().set_school_quota(
        school_id, monthly_requests, monthly_tokens, monthly_cost
    )
