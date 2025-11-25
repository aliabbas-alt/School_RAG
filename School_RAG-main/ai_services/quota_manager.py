# ai_services/quota_manager.py
"""Quota management for multi-tenant school system."""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QuotaManager:
    """
    Manage AI usage quotas per school.
    Prevents schools from exceeding their allocated limits.
    """
    
    def __init__(self):
        """Initialize quota manager."""
        self.school_quotas: Dict[str, Dict] = {}
        self.school_usage: Dict[str, Dict] = {}
    
    def set_quota(self, school_id: str, monthly_requests: int = 10000, 
                  monthly_tokens: int = 1000000, monthly_cost: float = 100.0):
        """
        Set quota for a school.
        
        Args:
            school_id: School identifier
            monthly_requests: Max requests per month
            monthly_tokens: Max tokens per month
            monthly_cost: Max cost per month (USD)
        """
        self.school_quotas[school_id] = {
            "requests": monthly_requests,
            "tokens": monthly_tokens,
            "cost": monthly_cost,
            "reset_date": self._get_next_reset_date()
        }
        
        self.school_usage[school_id] = {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0
        }
        
        logger.info(f"Quota set for {school_id}: {monthly_requests} req/month")
    
    def check_quota(self, school_id: str, estimated_tokens: int, 
                    estimated_cost: float) -> tuple[bool, str]:
        """
        Check if school has quota available.
        
        Returns:
            (allowed, reason) tuple
        """
        if school_id not in self.school_quotas:
            # No quota set = unlimited (for demo/testing)
            return True, "No quota restrictions"
        
        quota = self.school_quotas[school_id]
        usage = self.school_usage[school_id]
        
        # Check if reset needed
        if datetime.now() >= quota["reset_date"]:
            self._reset_usage(school_id)
            usage = self.school_usage[school_id]
        
        # Check request limit
        if usage["requests"] + 1 > quota["requests"]:
            return False, f"Monthly request limit exceeded ({quota['requests']})"
        
        # Check token limit
        if usage["tokens"] + estimated_tokens > quota["tokens"]:
            return False, f"Monthly token limit exceeded ({quota['tokens']})"
        
        # Check cost limit
        if usage["cost"] + estimated_cost > quota["cost"]:
            return False, f"Monthly cost limit exceeded (${quota['cost']})"
        
        return True, "Quota available"
    
    def record_usage(self, school_id: str, tokens: int, cost: float):
        """Record usage for a school."""
        if school_id not in self.school_usage:
            self.school_usage[school_id] = {"requests": 0, "tokens": 0, "cost": 0.0}
        
        usage = self.school_usage[school_id]
        usage["requests"] += 1
        usage["tokens"] += tokens
        usage["cost"] += cost
        
        logger.info(f"{school_id} usage: {usage['requests']} req, "
                   f"{usage['tokens']} tokens, ${usage['cost']:.2f}")
    
    def _get_next_reset_date(self) -> datetime:
        """Get next month's reset date."""
        now = datetime.now()
        if now.month == 12:
            return datetime(now.year + 1, 1, 1)
        else:
            return datetime(now.year, now.month + 1, 1)
    
    def _reset_usage(self, school_id: str):
        """Reset usage for new month."""
        self.school_usage[school_id] = {"requests": 0, "tokens": 0, "cost": 0.0}
        self.school_quotas[school_id]["reset_date"] = self._get_next_reset_date()
        logger.info(f"Usage reset for {school_id}")
    
    def get_usage_report(self, school_id: str) -> Dict:
        """Get detailed usage report for school."""
        if school_id not in self.school_quotas:
            return {"status": "No quota set"}
        
        quota = self.school_quotas[school_id]
        usage = self.school_usage[school_id]
        
        return {
            "school_id": school_id,
            "usage": usage,
            "quota": quota,
            "percentage_used": {
                "requests": (usage["requests"] / quota["requests"]) * 100,
                "tokens": (usage["tokens"] / quota["tokens"]) * 100,
                "cost": (usage["cost"] / quota["cost"]) * 100
            },
            "reset_date": quota["reset_date"].isoformat()
        }


# Global quota manager
_quota_manager = QuotaManager()


def get_quota_manager() -> QuotaManager:
    """Get global quota manager."""
    return _quota_manager
