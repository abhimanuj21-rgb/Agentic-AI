"""WebSearchTool for fetching external technical documentation and specs."""

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Type
try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None

try:
    from crewai.tools import BaseTool
except ImportError:
    class BaseTool:
        name: str = ""
        description: str = ""
        args_schema: Optional[Type[BaseModel]] = None

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def _run(self, *args, **kwargs):
            raise NotImplementedError


class WebSearchInput(BaseModel):
    """Input schema for WebSearchTool."""
    query: str = Field(..., description="The technical or coding query to search for online.")


class WebSearchTool(BaseTool):
    """CrewAI tool for external web search (Tavily / SerpAPI wrapper)."""
    name: str = "web_search_tool"
    description: str = (
        "Query the web for the latest programming libraries, documentation, "
        "APIs, error messages, and technical specifications. Returns authoritative "
        "URLs, titles, and content snippets."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    max_results: int = 3

    def _search_tavily(self, query: str, api_key: str) -> Optional[str]:
        """Perform search using Tavily API via standard library HTTP request."""
        try:
            url = "https://api.tavily.com/search"
            payload = json.dumps({
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": self.max_results,
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = data.get("results", [])
            if not results:
                return "STATUS: NO_WEB_RESULTS\nNo relevant web results were found."

            output = ["STATUS: WEB_SEARCH_SUCCESS\n"]
            if data.get("answer"):
                output.append(f"AI Overview: {data['answer']}\n")

            for i, res in enumerate(results, 1):
                title = res.get("title", "No Title")
                link = res.get("url", "No URL")
                snippet = res.get("content", "").strip()
                output.append(f"[{i}] {title}\nURL: {link}\nSnippet: {snippet}\n")

            return "\n".join(output)
        except Exception as e:
            return f"STATUS: TAVILY_ERROR\nError performing Tavily search: {str(e)}"

    def _search_serpapi(self, query: str, api_key: str) -> Optional[str]:
        """Perform search using SerpAPI."""
        try:
            params = urllib.parse.urlencode({
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": self.max_results,
            })
            url = f"https://serpapi.com/search?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "agentic-rag-coder/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            organic = data.get("organic_results", [])
            if not organic:
                return "STATUS: NO_WEB_RESULTS\nNo results from SerpAPI."

            output = ["STATUS: WEB_SEARCH_SUCCESS\n"]
            for i, res in enumerate(organic[: self.max_results], 1):
                title = res.get("title", "No Title")
                link = res.get("link", "No Link")
                snippet = res.get("snippet", "").strip()
                output.append(f"[{i}] {title}\nURL: {link}\nSnippet: {snippet}\n")

            return "\n".join(output)
        except Exception as e:
            return f"STATUS: SERPAPI_ERROR\nError querying SerpAPI: {str(e)}"

    def _run(self, query: str) -> str:
        """Route to available search provider or report guardrail status."""
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip()

        if tavily_key and not tavily_key.startswith("your_"):
            return self._search_tavily(query, tavily_key)
        elif serpapi_key and not serpapi_key.startswith("your_"):
            return self._search_serpapi(query, serpapi_key)
        else:
            return (
                "STATUS: WEB_SEARCH_UNAVAILABLE\n"
                "External web search is currently disabled because neither TAVILY_API_KEY "
                "nor SERPAPI_API_KEY is configured in the environment (.env). "
                "Please answer using local context or inform the user that external research is unavailable."
            )
