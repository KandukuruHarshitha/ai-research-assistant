import time
from functools import wraps
from logger import logger

class AIAssistantError(Exception):
    """Base exception for the application."""
    pass

class RateLimitError(AIAssistantError):
    """Raised when API rate limits are exceeded."""
    pass

class DatabaseError(AIAssistantError):
    """Raised when there are issues with the vector database."""
    pass

def retry_on_rate_limit(max_retries=3, initial_delay=5):
    """Decorator to retry a function if a rate limit error occurs."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "rate_limit" in err_msg or "429" in err_msg:
                        retries += 1
                        if retries > max_retries:
                            logger.error(f"Max retries reached for {func.__name__} after Rate Limit.")
                            raise RateLimitError("API rate limit exceeded after multiple retries.")
                        logger.warning(f"Rate limit hit in {func.__name__}. Retrying in {delay}s (Attempt {retries}/{max_retries})...")
                        time.sleep(delay)
                        delay *= 2 # Exponential backoff
                    else:
                        logger.error(f"Unexpected error in {func.__name__}: {e}")
                        raise e
            return None
        return wrapper
    return decorator
