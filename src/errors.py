"""Errors that must stop the process before it serves traffic."""


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""
