"""
Tests for the enhanced logging system.

Tests cover:
- configure_logging() function
- enable_debug() and disable_debug() functions
- get_logger() function
- is_reasoning_logging_enabled() function
- Environment variable configuration
- Different log formats
- Reasoning block logging
"""

import logging
import os
from io import StringIO

import pytest

from agentflow.state import Message
from agentflow.state.message_block import ReasoningBlock, TextBlock
from agentflow.utils.logging import (
    JSONFormatter,
    configure_logging,
    disable_debug,
    enable_debug,
    get_logger,
    is_reasoning_logging_enabled,
)


@pytest.fixture(autouse=True)
def cleanup_logging():
    """Clean up logging configuration after each test."""
    # Clean up before test
    import agentflow.utils.logging as logging_module
    logging_module._configured = False
    
    yield
    
    # Reset the agentflow logger
    logger = logging.getLogger("agentflow")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.NOTSET)
    
    # Reset the configured flag
    logging_module._configured = False

    # Clean up environment variables
    for key in [
        "AGENTFLOW_DEBUG",
        "AGENTFLOW_LOG_LEVEL",
        "AGENTFLOW_LOG_FORMAT",
        "AGENTFLOW_LOG_REASONING",
    ]:
        os.environ.pop(key, None)


def test_configure_logging_default():
    """Test configure_logging with default settings."""
    configure_logging()

    logger = logging.getLogger("agentflow")
    assert logger.level == logging.INFO
    assert len([h for h in logger.handlers if not isinstance(h, logging.NullHandler)]) > 0


def test_configure_logging_debug_level():
    """Test configure_logging with DEBUG level."""
    configure_logging(level=logging.DEBUG)

    logger = logging.getLogger("agentflow")
    # Check effective level (logger.level shows 0 if it inherits, but setLevel was called)
    assert logger.getEffectiveLevel() == logging.DEBUG or logger.level == logging.DEBUG


def test_configure_logging_string_level():
    """Test configure_logging with string level."""
    configure_logging(level="WARNING")

    logger = logging.getLogger("agentflow")
    assert logger.getEffectiveLevel() == logging.WARNING or logger.level == logging.WARNING


def test_configure_logging_formats():
    """Test different log formats."""
    # Simple format
    configure_logging(format_type="simple", force=True)
    logger = logging.getLogger("agentflow")
    handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
    assert len(handlers) > 0
    assert handlers[0].formatter is not None

    # Detailed format
    configure_logging(format_type="detailed", force=True)
    handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
    assert len(handlers) > 0

    # JSON format
    configure_logging(format_type="json", force=True)
    handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
    assert len(handlers) > 0
    assert isinstance(handlers[0].formatter, JSONFormatter)


def test_configure_logging_custom_handler():
    """Test configure_logging with custom handler."""
    stream = StringIO()
    custom_handler = logging.StreamHandler(stream)

    configure_logging(handler=custom_handler, level=logging.INFO)

    logger = get_logger()
    logger.info("Test message")
    
    # Flush the handler to ensure output is written
    custom_handler.flush()

    output = stream.getvalue()
    assert "Test message" in output


def test_configure_logging_force():
    """Test that force=True allows reconfiguration."""
    configure_logging(level=logging.INFO)
    logger = logging.getLogger("agentflow")
    assert logger.getEffectiveLevel() == logging.INFO or logger.level == logging.INFO

    # Without force, should not reconfigure
    configure_logging(level=logging.DEBUG)
    assert logger.getEffectiveLevel() == logging.INFO or logger.level == logging.INFO

    # With force, should reconfigure
    configure_logging(level=logging.DEBUG, force=True)
    assert logger.getEffectiveLevel() == logging.DEBUG or logger.level == logging.DEBUG


def test_enable_debug():
    """Test enable_debug() function."""
    enable_debug()

    logger = logging.getLogger("agentflow")
    assert logger.level == logging.DEBUG


def test_disable_debug():
    """Test disable_debug() function."""
    enable_debug()
    disable_debug()

    logger = logging.getLogger("agentflow")
    assert logger.level == logging.INFO


def test_get_logger_default():
    """Test get_logger() with default (no name)."""
    logger = get_logger()
    assert logger.name == "agentflow"


def test_get_logger_with_name():
    """Test get_logger() with custom name."""
    logger = get_logger("custom")
    assert logger.name == "agentflow.custom"


def test_is_reasoning_logging_enabled_default():
    """Test is_reasoning_logging_enabled() when not set."""
    assert is_reasoning_logging_enabled() is False


def test_is_reasoning_logging_enabled_true():
    """Test is_reasoning_logging_enabled() when enabled."""
    os.environ["AGENTFLOW_LOG_REASONING"] = "1"
    assert is_reasoning_logging_enabled() is True

    os.environ["AGENTFLOW_LOG_REASONING"] = "true"
    assert is_reasoning_logging_enabled() is True

    os.environ["AGENTFLOW_LOG_REASONING"] = "yes"
    assert is_reasoning_logging_enabled() is True


def test_is_reasoning_logging_enabled_false():
    """Test is_reasoning_logging_enabled() when disabled."""
    os.environ["AGENTFLOW_LOG_REASONING"] = "0"
    assert is_reasoning_logging_enabled() is False

    os.environ["AGENTFLOW_LOG_REASONING"] = "false"
    assert is_reasoning_logging_enabled() is False


def test_env_var_debug():
    """Test AGENTFLOW_DEBUG environment variable."""
    os.environ["AGENTFLOW_DEBUG"] = "1"
    configure_logging(force=True)

    logger = logging.getLogger("agentflow")
    assert logger.level == logging.DEBUG


def test_env_var_log_level():
    """Test AGENTFLOW_LOG_LEVEL environment variable."""
    os.environ["AGENTFLOW_LOG_LEVEL"] = "WARNING"
    configure_logging(force=True)

    logger = logging.getLogger("agentflow")
    assert logger.level == logging.WARNING


def test_env_var_log_format():
    """Test AGENTFLOW_LOG_FORMAT environment variable."""
    os.environ["AGENTFLOW_LOG_FORMAT"] = "json"
    configure_logging(force=True)

    logger = logging.getLogger("agentflow")
    handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
    assert len(handlers) > 0
    assert isinstance(handlers[0].formatter, JSONFormatter)


def test_json_formatter():
    """Test JSONFormatter output."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    output = formatter.format(record)
    assert '"level": "INFO"' in output
    assert '"message": "Test message"' in output
    assert '"logger": "test"' in output


def test_logging_with_actual_messages():
    """Test logging with actual Message objects containing reasoning."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    configure_logging(handler=handler, level=logging.INFO, force=True)

    # Enable reasoning logging
    os.environ["AGENTFLOW_LOG_REASONING"] = "1"

    # Create a message with reasoning
    message = Message(
        role="assistant",
        content=[
            ReasoningBlock(
                summary="Test reasoning summary",
                details=["Step 1", "Step 2"],
            ),
            TextBlock(text="Response text"),
        ],
    )

    # This would be logged by the invoke_handler in actual execution
    logger = get_logger("test")
    if is_reasoning_logging_enabled():
        for block in message.content:
            if isinstance(block, ReasoningBlock):
                logger.info("[REASONING] %s", block.summary)

    output = stream.getvalue()
    assert "[REASONING]" in output
    assert "Test reasoning summary" in output


def test_multiple_loggers():
    """Test that multiple loggers work correctly."""
    configure_logging(level=logging.DEBUG, force=True)

    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    assert logger1.name == "agentflow.module1"
    assert logger2.name == "agentflow.module2"

    # Both should inherit from agentflow logger
    assert logger1.level == logging.NOTSET  # Inherits from parent
    assert logger2.level == logging.NOTSET  # Inherits from parent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
