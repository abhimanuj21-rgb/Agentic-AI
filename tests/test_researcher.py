"""Standalone isolation test for WebSearchTool and Researcher Agent."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.web_search_tool import WebSearchTool


def test_web_search_tool_isolation():
    print("=" * 60)
    print("TESTING WEB SEARCH TOOL IN ISOLATION")
    print("=" * 60)

    tool = WebSearchTool(max_results=2)

    # 1. Test guardrail behavior when no API keys are provided
    orig_tavily = os.environ.get("TAVILY_API_KEY")
    orig_serpapi = os.environ.get("SERPAPI_API_KEY")

    try:
        # Simulate missing API keys
        if "TAVILY_API_KEY" in os.environ:
            del os.environ["TAVILY_API_KEY"]
        if "SERPAPI_API_KEY" in os.environ:
            del os.environ["SERPAPI_API_KEY"]

        output = tool._run("Latest features in Python 3.13")
        print("\n--- Guardrail Test (No Keys) Output ---")
        print(output)
        assert "WEB_SEARCH_UNAVAILABLE" in output, "Tool should report unavailable when keys are absent."

    finally:
        # Restore original keys
        if orig_tavily:
            os.environ["TAVILY_API_KEY"] = orig_tavily
        if orig_serpapi:
            os.environ["SERPAPI_API_KEY"] = orig_serpapi

    # 2. Test live search if key is provided
    if os.getenv("TAVILY_API_KEY") and not os.getenv("TAVILY_API_KEY").startswith("your_"):
        print("\n--- Live Tavily Search Test ---")
        live_output = tool._run("FastAPI dependency injection pattern")
        print(live_output[:300] + "...")
        assert "STATUS:" in live_output
    else:
        print("\n[INFO] Skipped live Tavily network call (no live TAVILY_API_KEY set).")

    print("\n[SUCCESS] WebSearchTool isolation test passed!\n")


if __name__ == "__main__":
    test_web_search_tool_isolation()
