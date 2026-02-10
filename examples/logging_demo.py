"""
Demonstration of Agentflow's Enhanced Logging System.

This example shows how to configure logging, enable debug output,
and capture reasoning blocks from agent execution.

Features demonstrated:
1. Basic logging configuration
2. Debug mode enabling/disabling
3. Different log formats (simple, detailed, json)
4. Reasoning capture logging
5. Environment variable configuration
"""

import asyncio
import logging
import os
from typing import Any

from agentflow import AgentState, StateGraph
from agentflow.state import Message
from agentflow.state.message_block import ReasoningBlock, TextBlock
from agentflow.utils import (
    configure_logging,
    disable_debug,
    enable_debug,
    get_logger,
    is_reasoning_logging_enabled,
)


# Example 1: Basic Logging Configuration
def example_basic_logging():
    """Demonstrate basic logging configuration."""
    print("\n=== Example 1: Basic Logging Configuration ===")

    # Configure with INFO level (default)
    configure_logging(level=logging.INFO, format_type="simple")

    logger = get_logger("demo")
    logger.info("This is an INFO message")
    logger.debug("This DEBUG message won't appear (INFO level)")

    # Enable debug mode
    enable_debug()
    logger.debug("Now this DEBUG message will appear")

    # Disable debug mode
    disable_debug()
    logger.debug("This DEBUG message won't appear again")

    print("✓ Basic logging configuration demonstrated\n")


# Example 2: Different Log Formats
def example_log_formats():
    """Demonstrate different log format options."""
    print("\n=== Example 2: Different Log Formats ===")

    logger = get_logger("format_demo")

    # Simple format
    print("\nSimple format:")
    configure_logging(level=logging.INFO, format_type="simple", force=True)
    logger.info("Simple format message")

    # Detailed format
    print("\nDetailed format:")
    configure_logging(level=logging.INFO, format_type="detailed", force=True)
    logger.info("Detailed format message")

    # JSON format
    print("\nJSON format:")
    configure_logging(level=logging.INFO, format_type="json", force=True)
    logger.info("JSON format message")

    print("\n✓ Log formats demonstrated\n")


# Example 3: Environment Variable Configuration
def example_env_configuration():
    """Demonstrate environment variable-based configuration."""
    print("\n=== Example 3: Environment Variable Configuration ===")

    # Set environment variables
    os.environ["AGENTFLOW_DEBUG"] = "1"
    os.environ["AGENTFLOW_LOG_FORMAT"] = "detailed"

    # Configure logging - it will read from environment variables
    configure_logging(force=True)

    logger = get_logger("env_demo")
    logger.debug("Debug enabled via AGENTFLOW_DEBUG environment variable")

    # Clean up
    del os.environ["AGENTFLOW_DEBUG"]
    del os.environ["AGENTFLOW_LOG_FORMAT"]

    print("✓ Environment variable configuration demonstrated\n")


# Example 4: Reasoning Capture
def example_reasoning_capture():
    """Demonstrate reasoning capture logging."""
    print("\n=== Example 4: Reasoning Capture ===")

    # Enable reasoning logging
    os.environ["AGENTFLOW_LOG_REASONING"] = "1"

    # Configure logging with debug level
    configure_logging(level=logging.DEBUG, format_type="detailed", force=True)

    logger = get_logger("reasoning_demo")

    # Check if reasoning logging is enabled
    if is_reasoning_logging_enabled():
        logger.info("Reasoning logging is ENABLED")
    else:
        logger.info("Reasoning logging is DISABLED")

    # Simulate reasoning block logging
    reasoning = ReasoningBlock(
        summary="Analyzing user query to determine best approach",
        details=[
            "Step 1: Parse user intent from message",
            "Step 2: Identify required tools and data",
            "Step 3: Plan execution sequence",
        ],
    )

    logger.info("[REASONING] %s", reasoning.summary)
    if reasoning.details:
        for idx, detail in enumerate(reasoning.details):
            logger.debug("[REASONING_DETAIL] Step %d: %s", idx, detail)

    # Clean up
    del os.environ["AGENTFLOW_LOG_REASONING"]

    print("✓ Reasoning capture demonstrated\n")


# Example 5: Graph Execution with Logging
async def example_graph_with_logging():
    """Demonstrate logging in actual graph execution."""
    print("\n=== Example 5: Graph Execution with Logging ===")

    # Enable full debug logging with reasoning
    os.environ["AGENTFLOW_LOG_REASONING"] = "1"
    configure_logging(level=logging.DEBUG, format_type="detailed", force=True)

    logger = get_logger("graph_demo")
    logger.info("Starting graph execution with enhanced logging")

    # Create a simple graph
    def reasoning_node(state: AgentState, config: dict[str, Any]) -> list[Message]:
        """Node that returns a message with reasoning."""
        logger.info("Executing reasoning_node")

        # Create a message with reasoning block
        message = Message(
            role="assistant",
            content=[
                ReasoningBlock(
                    summary="Analyzed the input and formulated response",
                    details=[
                        "Identified user intent",
                        "Selected appropriate response strategy",
                        "Composed structured reply",
                    ],
                ),
                TextBlock(text="Hello! I've analyzed your message and am ready to help."),
            ],
        )

        logger.debug("Node returning message with reasoning")
        return [message]

    # Build graph
    graph = StateGraph()
    graph.add_node("reasoning", reasoning_node)
    graph.set_entry_point("reasoning")
    graph.add_edge("reasoning", "__end__")

    # Compile and execute
    compiled = graph.compile()

    input_data = {"messages": [Message.text_message("Hello")]}
    result = await compiled.ainvoke(input_data)

    logger.info("Graph execution completed")
    logger.debug("Result state has %d messages", len(result.context))

    # Clean up
    await compiled.aclose()
    del os.environ["AGENTFLOW_LOG_REASONING"]

    print("✓ Graph execution with logging demonstrated\n")


# Example 6: Logging Best Practices
def example_best_practices():
    """Demonstrate logging best practices."""
    print("\n=== Example 6: Logging Best Practices ===")

    configure_logging(level=logging.INFO, format_type="detailed", force=True)

    logger = get_logger("best_practices")

    # 1. Use appropriate log levels
    logger.debug("Detailed debug information (use sparingly)")
    logger.info("General informational message")
    logger.warning("Warning about something that might need attention")
    logger.error("Error that occurred but execution continues")

    # 2. Use structured logging with context
    user_id = "user123"
    action = "login"
    logger.info("User action: user_id=%s, action=%s", user_id, action)

    # 3. Log exceptions properly
    try:
        result = 1 / 0
    except ZeroDivisionError:
        logger.exception("Error during calculation")

    # 4. Use module-specific loggers
    module_logger = get_logger("my_module")
    module_logger.info("Message from specific module")

    print("✓ Best practices demonstrated\n")


# Main execution
def main():
    """Run all examples."""
    print("=" * 70)
    print("AGENTFLOW ENHANCED LOGGING DEMONSTRATION")
    print("=" * 70)

    # Run synchronous examples
    example_basic_logging()
    example_log_formats()
    example_env_configuration()
    example_reasoning_capture()
    example_best_practices()

    # Run async example
    asyncio.run(example_graph_with_logging())

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
