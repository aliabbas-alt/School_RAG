# ai_services/rate_limiter.py
"""
Rate Limiter - Token Bucket Algorithm

Prevents exceeding provider rate limits for both requests and tokens.
Uses sliding window approach for accurate rate limiting.

Features:
- Dual limiting (requests per minute AND tokens per minute)
- Sliding window for accurate tracking
- Thread-safe operation
- Automatic cleanup of old entries
- Detailed metrics
- Configurable wait behavior
"""

import time
import logging
from collections import deque
from threading import Lock
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    requests_per_minute: int
    tokens_per_minute: int
    requests_per_day: Optional[int] = None
    tokens_per_day: Optional[int] = None


class RateLimiter:
    """
    Token bucket rate limiter with sliding window.
    
    Prevents exceeding provider rate limits by tracking:
    - Requests per minute
    - Tokens per minute
    - Optional daily limits
    
    Thread-safe and uses sliding window for accurate tracking.
    """
    
    def __init__(
        self, 
        requests_per_minute: int = 60, 
        tokens_per_minute: int = 90000,
        requests_per_day: Optional[int] = None,
        tokens_per_day: Optional[int] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute
            tokens_per_minute: Max tokens per minute
            requests_per_day: Optional daily request limit
            tokens_per_day: Optional daily token limit
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_day = requests_per_day
        self.tokens_per_day = tokens_per_day
        
        # Sliding window tracking (minute-level)
        self.request_timestamps = deque()  # (timestamp, )
        self.token_usage = deque()  # (timestamp, tokens)
        
        # Daily tracking
        self.daily_requests = 0
        self.daily_tokens = 0
        self.daily_reset_time = self._get_next_day()
        
        # Thread safety
        self.lock = Lock()
        
        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_wait_time = 0.0
        self.rate_limit_hits = 0
        self.last_limit_hit: Optional[datetime] = None
        
        logger.info(f"Rate limiter initialized: {requests_per_minute} req/min, "
                   f"{tokens_per_minute} tokens/min")
    
    def acquire(
        self, 
        estimated_tokens: int = 1000,
        wait: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Try to acquire permission for a request.
        
        Args:
            estimated_tokens: Estimated tokens for this request
            wait: If True, block until allowed; if False, return immediately
        
        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        with self.lock:
            now = time.time()
            
            # Clean old entries
            self._clean_old_entries(now)
            
            # Check daily limits first (if configured)
            if not self._check_daily_limits(estimated_tokens):
                reason = self._get_limit_reason(estimated_tokens)
                self.rate_limit_hits += 1
                self.last_limit_hit = datetime.now()
                
                if not wait:
                    logger.warning(f"⏸️  Rate limit: {reason}")
                    return False, reason
                
                # Wait for daily reset
                wait_time = self._wait_for_daily_reset()
                self.total_wait_time += wait_time
                return self.acquire(estimated_tokens, wait=False)
            
            # Check per-minute limits
            if not self._check_minute_limits(estimated_tokens):
                reason = self._get_limit_reason(estimated_tokens)
                self.rate_limit_hits += 1
                self.last_limit_hit = datetime.now()
                
                if not wait:
                    logger.warning(f"⏸️  Rate limit: {reason}")
                    return False, reason
                
                # Wait for capacity
                wait_time = self._wait_for_capacity(estimated_tokens)
                self.total_wait_time += wait_time
                return self.acquire(estimated_tokens, wait=False)
            
            # All checks passed - allow request
            self._record_request(now, estimated_tokens)
            
            return True, None
    
    def wait_if_needed(self, estimated_tokens: int = 1000):
        """
        Block until request is allowed.
        Convenience method that always waits.
        
        Args:
            estimated_tokens: Estimated tokens for this request
        """
        allowed, reason = self.acquire(estimated_tokens, wait=True)
        if not allowed:
            # Should never happen with wait=True, but just in case
            raise Exception(f"Rate limit wait failed: {reason}")
    
    def try_acquire(self, estimated_tokens: int = 1000) -> bool:
        """
        Try to acquire without waiting.
        
        Args:
            estimated_tokens: Estimated tokens
        
        Returns:
            True if allowed, False if rate limited
        """
        allowed, _ = self.acquire(estimated_tokens, wait=False)
        return allowed
    
    def _check_minute_limits(self, estimated_tokens: int) -> bool:
        """Check per-minute limits."""
        # Check request rate
        if len(self.request_timestamps) >= self.requests_per_minute:
            return False
        
        # Check token rate
        total_tokens = sum(tokens for _, tokens in self.token_usage)
        if total_tokens + estimated_tokens > self.tokens_per_minute:
            return False
        
        return True
    
    def _check_daily_limits(self, estimated_tokens: int) -> bool:
        """Check daily limits (if configured)."""
        # Reset daily counters if needed
        if time.time() >= self.daily_reset_time:
            self._reset_daily_counters()
        
        # Check daily request limit
        if self.requests_per_day is not None:
            if self.daily_requests >= self.requests_per_day:
                return False
        
        # Check daily token limit
        if self.tokens_per_day is not None:
            if self.daily_tokens + estimated_tokens > self.tokens_per_day:
                return False
        
        return True
    
    def _record_request(self, now: float, tokens: int):
        """Record a request in the sliding window."""
        # Minute-level tracking
        self.request_timestamps.append(now)
        self.token_usage.append((now, tokens))
        
        # Daily tracking
        self.daily_requests += 1
        self.daily_tokens += tokens
        
        # Overall metrics
        self.total_requests += 1
        self.total_tokens += tokens
    
    def _clean_old_entries(self, now: float):
        """Remove entries older than 1 minute."""
        cutoff = now - 60  # 60 seconds = 1 minute
        
        # Clean request timestamps
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
        
        # Clean token usage
        while self.token_usage and self.token_usage[0][0] < cutoff:
            self.token_usage.popleft()
    
    def _wait_for_capacity(self, estimated_tokens: int) -> float:
        """
        Wait until there's capacity in the sliding window.
        
        Returns:
            Time waited in seconds
        """
        start_wait = time.time()
        
        while True:
            now = time.time()
            self._clean_old_entries(now)
            
            if self._check_minute_limits(estimated_tokens):
                wait_time = time.time() - start_wait
                logger.info(f"✅ Rate limit cleared after {wait_time:.1f}s")
                return wait_time
            
            # Wait 1 second before checking again
            time.sleep(1)
    
    def _wait_for_daily_reset(self) -> float:
        """Wait until daily reset time."""
        start_wait = time.time()
        wait_seconds = self.daily_reset_time - time.time()
        
        if wait_seconds > 0:
            logger.warning(f"⏳ Waiting {wait_seconds:.0f}s for daily limit reset...")
            time.sleep(wait_seconds)
        
        self._reset_daily_counters()
        return time.time() - start_wait
    
    def _reset_daily_counters(self):
        """Reset daily usage counters."""
        self.daily_requests = 0
        self.daily_tokens = 0
        self.daily_reset_time = self._get_next_day()
        logger.info("🔄 Daily rate limits reset")
    
    def _get_next_day(self) -> float:
        """Get timestamp for next day (midnight)."""
        from datetime import datetime, timedelta
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        midnight = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.timestamp()
    
    def _get_limit_reason(self, estimated_tokens: int) -> str:
        """Get human-readable reason for rate limit."""
        reasons = []
        
        # Check minute limits
        if len(self.request_timestamps) >= self.requests_per_minute:
            reasons.append(f"Request limit ({self.requests_per_minute}/min)")
        
        total_tokens = sum(tokens for _, tokens in self.token_usage)
        if total_tokens + estimated_tokens > self.tokens_per_minute:
            reasons.append(f"Token limit ({self.tokens_per_minute}/min)")
        
        # Check daily limits
        if self.requests_per_day and self.daily_requests >= self.requests_per_day:
            reasons.append(f"Daily request limit ({self.requests_per_day}/day)")
        
        if self.tokens_per_day and self.daily_tokens + estimated_tokens > self.tokens_per_day:
            reasons.append(f"Daily token limit ({self.tokens_per_day}/day)")
        
        return " | ".join(reasons) if reasons else "Rate limit exceeded"
    
    def get_current_usage(self) -> Dict[str, Any]:
        """
        Get current usage in sliding window.
        
        Returns:
            Dictionary with current usage stats
        """
        with self.lock:
            now = time.time()
            self._clean_old_entries(now)
            
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            
            return {
                "requests_last_minute": len(self.request_timestamps),
                "tokens_last_minute": current_tokens,
                "daily_requests": self.daily_requests,
                "daily_tokens": self.daily_tokens,
                "requests_remaining_minute": max(0, self.requests_per_minute - len(self.request_timestamps)),
                "tokens_remaining_minute": max(0, self.tokens_per_minute - current_tokens),
                "requests_remaining_day": max(0, (self.requests_per_day or float('inf')) - self.daily_requests) if self.requests_per_day else None,
                "tokens_remaining_day": max(0, (self.tokens_per_day or float('inf')) - self.daily_tokens) if self.tokens_per_day else None
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive rate limiter metrics.
        
        Returns:
            Dictionary with all metrics
        """
        usage = self.get_current_usage()
        
        return {
            "limits": {
                "requests_per_minute": self.requests_per_minute,
                "tokens_per_minute": self.tokens_per_minute,
                "requests_per_day": self.requests_per_day,
                "tokens_per_day": self.tokens_per_day
            },
            "current_usage": usage,
            "all_time_metrics": {
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "rate_limit_hits": self.rate_limit_hits,
                "total_wait_time": round(self.total_wait_time, 2),
                "last_limit_hit": self.last_limit_hit.isoformat() if self.last_limit_hit else None
            },
            "utilization": {
                "requests": (usage["requests_last_minute"] / self.requests_per_minute) * 100,
                "tokens": (usage["tokens_last_minute"] / self.tokens_per_minute) * 100
            }
        }
    
    def reset_metrics(self):
        """Reset metric counters (not limits or usage)."""
        with self.lock:
            self.total_requests = 0
            self.total_tokens = 0
            self.total_wait_time = 0.0
            self.rate_limit_hits = 0
            self.last_limit_hit = None
            logger.info("Rate limiter metrics reset")
    
    def update_limits(
        self, 
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        tokens_per_day: Optional[int] = None
    ):
        """
        Update rate limits dynamically.
        
        Args:
            requests_per_minute: New request limit
            tokens_per_minute: New token limit
            requests_per_day: New daily request limit
            tokens_per_day: New daily token limit
        """
        with self.lock:
            if requests_per_minute is not None:
                self.requests_per_minute = requests_per_minute
            if tokens_per_minute is not None:
                self.tokens_per_minute = tokens_per_minute
            if requests_per_day is not None:
                self.requests_per_day = requests_per_day
            if tokens_per_day is not None:
                self.tokens_per_day = tokens_per_day
            
            logger.info(f"Rate limits updated: {self.requests_per_minute} req/min, "
                       f"{self.tokens_per_minute} tokens/min")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"RateLimiter(requests_per_minute={self.requests_per_minute}, "
                f"tokens_per_minute={self.tokens_per_minute})")
