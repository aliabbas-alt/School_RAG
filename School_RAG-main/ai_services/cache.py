# ai_services/cache.py
"""Response caching to reduce API calls and costs."""

import hashlib
import json
import time
import logging
from typing import Dict, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    LRU cache for AI responses.
    Avoids repeated identical requests.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached responses
            ttl_seconds: Time-to-live for cached items (default 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict] = {}
        
        # Metrics
        self.hits = 0
        self.misses = 0
    
    def _create_key(self, request) -> str:
        """Create cache key from request."""
        key_data = {
            "prompt": request.prompt,
            "context": request.context,
            "model": request.parameters.get("model"),
            "temperature": request.parameters.get("temperature")
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, request) -> Optional[Dict]:
        """Get cached response if available and fresh."""
        key = self._create_key(request)
        
        if key in self.cache:
            cached = self.cache[key]
            age = time.time() - cached["timestamp"]
            
            if age < self.ttl_seconds:
                self.hits += 1
                logger.info(f"Cache HIT (age: {age:.1f}s)")
                return cached["response"]
            else:
                # Expired
                del self.cache[key]
        
        self.misses += 1
        logger.info("Cache MISS")
        return None
    
    def set(self, request, response):
        """Store response in cache."""
        key = self._create_key(request)
        
        # Evict oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
        logger.info(f"Cached response (total: {len(self.cache)})")
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_metrics(self) -> Dict:
        """Get cache metrics."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(total, 1),
            "size": len(self.cache),
            "max_size": self.max_size
        }


# Global cache instance
_cache = ResponseCache()


def get_cache() -> ResponseCache:
    """Get global cache instance."""
    return _cache
