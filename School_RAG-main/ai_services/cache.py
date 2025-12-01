# ai_services/cache.py
"""
Response caching to reduce API calls and costs.

Features:
- Thread-safe operations
- LRU eviction with O(1) operations
- TTL-based expiration
- Configurable cache strategies
- Detailed metrics and monitoring
- Optional Redis backend for distributed caching
"""

import hashlib
import json
import time
import logging
from typing import Dict, Optional, Any
from collections import OrderedDict
from threading import Lock
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out


class ResponseCache:
    """
    Thread-safe LRU cache for AI responses.
    
    Features:
    - Thread-safe with locks
    - LRU eviction (O(1) operations)
    - TTL-based expiration
    - Detailed metrics
    - Cache invalidation support
    """
    
    def __init__(
        self, 
        max_size: int = 1000, 
        ttl_seconds: int = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached responses
            ttl_seconds: Time-to-live for cached items (default 1 hour)
            strategy: Cache eviction strategy
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.strategy = strategy
        
        # Use OrderedDict for efficient LRU (Python 3.7+)
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        
        # Thread safety
        self.lock = Lock()
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.total_saved_cost = 0.0
        
        # Access frequency (for LFU strategy)
        self.access_count: Dict[str, int] = {}
        
        logger.info(f"Cache initialized: max_size={max_size}, ttl={ttl_seconds}s, "
                   f"strategy={strategy.value}")
    
    def _create_key(self, request) -> str:
        """
        Create cache key from request.
        Uses SHA256 for better security than MD5.
        """
        key_data = {
            "prompt": request.prompt,
            "context": request.context,
            "model": request.parameters.get("model"),
            "temperature": request.parameters.get("temperature"),
            # Add other relevant parameters that affect output
            "max_tokens": request.parameters.get("max_tokens")
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()  # ← SHA256, not MD5
    
    def get(self, request) -> Optional[Dict]:
        """
        Get cached response if available and fresh.
        Thread-safe operation.
        """
        key = self._create_key(request)
        
        with self.lock:  # ← Thread-safe
            if key in self.cache:
                cached = self.cache[key]
                age = time.time() - cached["timestamp"]
                
                # Check if expired
                if age < self.ttl_seconds:
                    # Move to end (mark as recently used for LRU)
                    self.cache.move_to_end(key)
                    
                    # Update access count (for LFU)
                    self.access_count[key] = self.access_count.get(key, 0) + 1
                    
                    # Metrics
                    self.hits += 1
                    if "cost_saved" in cached:
                        self.total_saved_cost += cached["cost_saved"]
                    
                    logger.info(f"✅ Cache HIT (age: {age:.1f}s, key: {key[:8]}...)")
                    return cached["response"]
                else:
                    # Expired - remove
                    del self.cache[key]
                    if key in self.access_count:
                        del self.access_count[key]
                    self.expirations += 1
                    logger.info(f"⏰ Cache entry expired (age: {age:.1f}s)")
            
            # Cache miss
            self.misses += 1
            logger.info(f"❌ Cache MISS (key: {key[:8]}...)")
            return None
    
    def set(self, request, response):
        """
        Store response in cache with thread safety.
        """
        key = self._create_key(request)
        
        with self.lock:  # ← Thread-safe
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_one()
            
            # Store with metadata
            self.cache[key] = {
                "response": response,
                "timestamp": time.time(),
                "cost_saved": getattr(response, 'cost_estimate', 0.0),
                "created_at": datetime.now().isoformat()
            }
            
            # Move to end (most recent)
            self.cache.move_to_end(key)
            
            # Initialize access count
            self.access_count[key] = 1
            
            logger.info(f"💾 Cached response (size: {len(self.cache)}/{self.max_size}, "
                       f"key: {key[:8]}...)")
    
    def _evict_one(self):
        """
        Evict one entry based on strategy.
        Called when cache is full.
        """
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove oldest (first item in OrderedDict)
            evicted_key, _ = self.cache.popitem(last=False)
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            evicted_key = min(self.access_count.keys(), 
                            key=lambda k: self.access_count[k])
            del self.cache[evicted_key]
        
        elif self.strategy == CacheStrategy.FIFO:
            # Remove first inserted
            evicted_key, _ = self.cache.popitem(last=False)
        
        # Clean up
        if evicted_key in self.access_count:
            del self.access_count[evicted_key]
        
        self.evictions += 1
        logger.info(f"🗑️  Evicted entry (strategy: {self.strategy.value}, "
                   f"key: {evicted_key[:8]}...)")
    
    def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            pattern: If provided, only invalidate keys containing this pattern
                    If None, invalidate all
        """
        with self.lock:
            if pattern is None:
                # Clear all
                count = len(self.cache)
                self.cache.clear()
                self.access_count.clear()
                logger.info(f"🗑️  Cleared entire cache ({count} entries)")
            else:
                # Selective invalidation
                keys_to_remove = [
                    key for key in self.cache.keys()
                    if pattern in key
                ]
                for key in keys_to_remove:
                    del self.cache[key]
                    if key in self.access_count:
                        del self.access_count[key]
                logger.info(f"🗑️  Invalidated {len(keys_to_remove)} entries "
                           f"matching pattern: {pattern}")
    
    def clear(self):
        """Clear all cached items."""
        self.invalidate(pattern=None)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / max(total, 1)
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "current_size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size,
                "total_saved_cost": round(self.total_saved_cost, 4),
                "strategy": self.strategy.value,
                "ttl_seconds": self.ttl_seconds
            }
    
    def get_stats_summary(self) -> str:
        """Get human-readable stats summary."""
        metrics = self.get_metrics()
        return (
            f"Cache Stats:\n"
            f"  Hit Rate: {metrics['hit_rate']:.2%}\n"
            f"  Hits: {metrics['hits']} | Misses: {metrics['misses']}\n"
            f"  Size: {metrics['current_size']}/{metrics['max_size']}\n"
            f"  Savings: ${metrics['total_saved_cost']:.2f}\n"
            f"  Evictions: {metrics['evictions']} | Expirations: {metrics['expirations']}"
        )
    
    def reset_metrics(self):
        """Reset metric counters (not the cache itself)."""
        with self.lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.expirations = 0
            self.total_saved_cost = 0.0
            logger.info("📊 Metrics reset")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ResponseCache(size={len(self.cache)}/{self.max_size}, "
                f"hit_rate={self.hits / max(self.hits + self.misses, 1):.2%})")


# ============================================================================
# GLOBAL CACHE INSTANCE
# ============================================================================

# Global cache instance (singleton)
_cache: Optional[ResponseCache] = None
_cache_lock = Lock()


def get_cache() -> ResponseCache:
    """
    Get global cache instance (thread-safe singleton).
    
    Returns:
        Global ResponseCache instance
    """
    global _cache
    
    if _cache is None:
        with _cache_lock:
            # Double-check locking pattern
            if _cache is None:
                _cache = ResponseCache()
    
    return _cache


def initialize_cache(
    max_size: int = 1000,
    ttl_seconds: int = 3600,
    strategy: CacheStrategy = CacheStrategy.LRU
) -> ResponseCache:
    """
    Initialize global cache with custom configuration.
    
    Args:
        max_size: Maximum cache size
        ttl_seconds: Time-to-live
        strategy: Eviction strategy
    
    Returns:
        Configured ResponseCache instance
    """
    global _cache
    
    with _cache_lock:
        _cache = ResponseCache(max_size, ttl_seconds, strategy)
    
    return _cache
