"""Provider failures, independent of which model served the call."""


class LLMError(Exception):
    """A model call failed. `retryable` tells the caller whether to try again."""

    retryable = False


class LLMTimeoutError(LLMError):
    retryable = True


class RateLimitError(LLMError):
    retryable = True


class AuthError(LLMError):
    retryable = False


class ContextOverflowError(LLMError):
    retryable = False


class MalformedResponseError(LLMError):
    retryable = False
