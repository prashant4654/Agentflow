"""
Logging utilities for Agentflow.

This module provides logging support for the Agentflow library following Python
logging best practices for library code.

By default, Agentflow uses a NullHandler to prevent "No handlers could be found"
warnings. Users can configure logging by getting the logger and adding their own
handlers.

Library Usage (within agentflow modules):
    Each module should create its own logger:

    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("This is an info message")

User Configuration Example:
    Users of the Agentflow library can configure logging like this::

        import logging
        from agentflow.utils.logging import configure_logging

        # Quick setup with defaults
        configure_logging(level=logging.DEBUG)

        # Or manually configure
        logger = logging.getLogger("agentflow")
        logger.setLevel(logging.DEBUG)

        # Add a handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

Environment Variables:
    - AGENTFLOW_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - AGENTFLOW_DEBUG: Set to "1", "true", "yes" to enable DEBUG level
    - AGENTFLOW_LOG_FORMAT: Set log format ("simple", "detailed", "json")
    - AGENTFLOW_LOG_REASONING: Set to "1", "true", "yes" to enable reasoning capture logs

Best Practices:
    - Library code should NEVER configure the root logger
    - Library code should NEVER add handlers except NullHandler
    - Library code should use module-level loggers (logging.getLogger(__name__))
    - Users control logging configuration in their applications

References:
    https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
"""

import json
import logging
import os
import sys
from typing import Literal


# Create the main agentflow logger
logger = logging.getLogger("agentflow")

# Add NullHandler by default to prevent "No handlers found" warnings
# Users can configure their own handlers as needed
logger.addHandler(logging.NullHandler())

# Track if logging has been configured
_configured = False


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ):
                log_data[key] = value

        return json.dumps(log_data)


def _parse_env_bool(value: str | None) -> bool:
    """Parse environment variable as boolean."""
    if value is None:
        return False
    return value.lower() in ("1", "true", "yes", "on")


def _get_log_level_from_env() -> int | None:
    """Get log level from environment variables."""
    # Check AGENTFLOW_DEBUG first
    if _parse_env_bool(os.getenv("AGENTFLOW_DEBUG")):
        return logging.DEBUG

    # Check AGENTFLOW_LOG_LEVEL
    level_str = os.getenv("AGENTFLOW_LOG_LEVEL")
    if level_str:
        level_str = level_str.upper()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return level_map.get(level_str)

    return None


def _get_log_format_from_env() -> str:
    """Get log format from environment variables."""
    format_type = os.getenv("AGENTFLOW_LOG_FORMAT", "simple").lower()
    return format_type


def configure_logging(
    level: int | str | None = None,
    format_type: Literal["simple", "detailed", "json"] = "simple",
    handler: logging.Handler | None = None,
    force: bool = False,
) -> None:
    """
    Configure logging for Agentflow.

    This is a convenience function for users to quickly set up logging.
    By default, it will only configure logging once unless force=True.

    Args:
        level: Logging level (e.g., logging.DEBUG, "DEBUG", logging.INFO).
               If None, checks environment variables, defaults to INFO.
        format_type: Log format type - "simple", "detailed", or "json".
        handler: Custom handler to use. If None, uses StreamHandler.
        force: Force reconfiguration even if already configured.

    Environment Variables:
        AGENTFLOW_DEBUG: Set to "1", "true", "yes" to enable DEBUG level
        AGENTFLOW_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        AGENTFLOW_LOG_FORMAT: Set format ("simple", "detailed", "json")

    Example:
        >>> from agentflow.utils.logging import configure_logging
        >>> import logging
        >>> configure_logging(level=logging.DEBUG)
        >>> configure_logging(format_type="json")
        >>> configure_logging(level="DEBUG", format_type="detailed")
    """
    global _configured

    if _configured and not force:
        return

    # Remove existing handlers from agentflow logger
    agentflow_logger = logging.getLogger("agentflow")
    for h in agentflow_logger.handlers[:]:
        if not isinstance(h, logging.NullHandler):
            agentflow_logger.removeHandler(h)

    # Determine log level
    if level is None:
        level = _get_log_level_from_env()
    if level is None:
        level = logging.INFO

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    agentflow_logger.setLevel(level)

    # Get format from environment if not specified
    if format_type == "simple" and os.getenv("AGENTFLOW_LOG_FORMAT"):
        format_type = _get_log_format_from_env()  # type: ignore

    # Create handler
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    if format_type == "json":
        formatter = JSONFormatter()
    elif format_type == "detailed":
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:  # simple
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    agentflow_logger.addHandler(handler)

    _configured = True

    # Log configuration
    agentflow_logger.debug(
        "Logging configured: level=%s, format=%s",
        logging.getLevelName(level),
        format_type,
    )


def enable_debug() -> None:
    """
    Enable DEBUG level logging with detailed format.

    This is a convenience function equivalent to:
        configure_logging(level=logging.DEBUG, format_type="detailed", force=True)

    Example:
        >>> from agentflow.utils.logging import enable_debug
        >>> enable_debug()
    """
    configure_logging(level=logging.DEBUG, format_type="detailed", force=True)


def disable_debug() -> None:
    """
    Disable DEBUG level logging, set to INFO level.

    Example:
        >>> from agentflow.utils.logging import disable_debug
        >>> disable_debug()
    """
    configure_logging(level=logging.INFO, force=True)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger for the specified name.

    Args:
        name: Logger name. If None, returns the main agentflow logger.
              If provided, returns a child logger under agentflow namespace.

    Returns:
        Logger instance.

    Example:
        >>> from agentflow.utils.logging import get_logger
        >>> logger = get_logger("my_module")
        >>> logger.info("Hello")
    """
    if name is None:
        return logger
    return logging.getLogger(f"agentflow.{name}")


def is_reasoning_logging_enabled() -> bool:
    """
    Check if reasoning logging is enabled via environment variable.

    Returns:
        True if reasoning logging is enabled, False otherwise.

    Example:
        >>> from agentflow.utils.logging import is_reasoning_logging_enabled
        >>> if is_reasoning_logging_enabled():
        ...     logger.debug("Reasoning: %s", reasoning_summary)
    """
    return _parse_env_bool(os.getenv("AGENTFLOW_LOG_REASONING"))


__all__ = [
    "logger",
    "configure_logging",
    "enable_debug",
    "disable_debug",
    "get_logger",
    "is_reasoning_logging_enabled",
    "JSONFormatter",
]
