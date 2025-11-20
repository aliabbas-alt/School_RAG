"""Custom exceptions for AI services."""


class AIServiceException(Exception):
    """Base exception for AI service errors."""
    pass


class AIServiceUnavailable(AIServiceException):
    """Raised when AI service is unavailable."""
    pass


class AIRateLimitExceeded(AIServiceException):
    """Raised when rate limit is exceeded."""
    pass


class AIInvalidResponse(AIServiceException):
    """Raised when response is invalid or malformed."""
    pass


class AIAuthenticationError(AIServiceException):
    """Raised when authentication fails."""
    pass


class AIQuotaExceeded(AIServiceException):
    """Raised when school's quota is exceeded."""
    pass
