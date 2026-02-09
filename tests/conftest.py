"""Pytest configuration and fixtures for all tests.

This module provides common fixtures and setup for the entire test suite.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up environment variables for testing.
    
    This fixture automatically runs for all test sessions and sets
    dummy API keys to prevent test failures due to missing credentials.
    This is test-only setup and does not affect production code.
    """
    # Set dummy OpenAI API key for tests
    # Using a valid-looking but fake key that won't make actual API calls
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-testing-only")
    
    # Set dummy Google API key for tests
    os.environ.setdefault("GEMINI_API_KEY", "dummy-gemini-key-for-testing-only")
    
    yield
    
    # Note: We don't clean up the environment variables since they're test-only
    # and won't affect any other processes
