"""External tools for the research agent.

For the MVP, the only tool is Tavily web search. The wrapper below converts
Tavily's response into a small, predictable shape the agent can use.
"""

from typing import Any

from tavily import TavilyClient


class SearchError(Exception):
    """Raised when a web search API call fails."""


def search_web(
    query: str,
    api_key: str,
    max_results: int = 5,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Search the web with Tavily and return normalized result dictionaries."""

    query = query.strip()
    if not query:
        return []

    tavily_client = client or TavilyClient(api_key=api_key)

    try:
        response = tavily_client.search(query=query, max_results=max_results)
    except Exception as exc:
        raise SearchError(f"Search failed for query '{query}': {exc}") from exc

    raw_results = response.get("results", []) if isinstance(response, dict) else []
    normalized = []

    for item in raw_results:
        if not isinstance(item, dict):
            continue

        url = str(item.get("url", "")).strip()
        if not url:
            continue

        normalized.append(
            {
                "title": str(item.get("title", "")).strip() or "Untitled",
                "url": url,
                "content": str(item.get("content", "")).strip(),
                "score": item.get("score", 0),
            }
        )

    return normalized
