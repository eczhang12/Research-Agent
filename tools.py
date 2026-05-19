"""External tools for the research agent.

AI agents often need tools. A tool is normal code that lets the agent do
something the language model cannot do by itself. In this project, the tool is
web search through the Tavily API.

The language model creates search queries, but Python makes the actual API
request to Tavily and normalizes the response into a simple shape.
"""

from typing import Any

from tavily import TavilyClient


class SearchError(Exception):
    """Explain web-search failures in a clear way.

    What this class does:
    - Represents errors from the Tavily search API.

    Why it exists:
    - `agent.py` can catch `SearchError` and return a friendly message instead
      of crashing the terminal session.

    Inputs:
    - A normal exception message string.

    Outputs:
    - No return value. Raising this exception interrupts normal execution until
      error-handling code catches it.
    """


def search_web(
    query: str,
    api_key: str,
    max_results: int = 5,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Search the web with Tavily and return normalized result dictionaries.

    What this function does:
    - Sends one search query to Tavily.
    - Reads Tavily's response.
    - Converts each result into a simple dictionary with the same keys every
      time: `title`, `url`, `content`, and `score`.

    Why it exists:
    - API responses often contain many fields and provider-specific details.
      Normalizing the response here keeps `agent.py` beginner-friendly because
      the agent can work with a predictable source format.

    Inputs:
    - `query`: the web search text.
    - `api_key`: secret Tavily API key from `.env`.
    - `max_results`: maximum number of results Tavily should return.
    - `client`: optional Tavily-like client for tests. This is dependency
      injection, and it avoids real network calls during testing.

    Outputs:
    - A list of dictionaries, for example:
      `{"title": "Example", "url": "https://...", "content": "...", "score": 0.8}`

    Example flow:
    - Agent asks to search `"Python AI research agent tutorial"`
    - Tavily returns raw API data
    - This function returns clean source dictionaries for the agent
    """

    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    # In real app usage, we create a Tavily client with the user's API key. In
    # tests, a fake client is passed in so tests stay fast, deterministic, and
    # free of external API calls.
    tavily_client = client or TavilyClient(api_key=api_key)

    try:
        raw_response = tavily_client.search(
            query=cleaned_query,
            max_results=max_results,
        )
    except Exception as exc:
        # Wrap provider-specific errors in our own SearchError. The rest of the
        # app only needs to understand "search failed", not every possible
        # Tavily/network exception type.
        raise SearchError(f"Search failed for query '{cleaned_query}': {exc}") from exc

    # Tavily returns a dictionary with a `results` list. This defensive check
    # keeps the app from crashing if a fake client or future API response has an
    # unexpected shape.
    raw_results = raw_response.get("results", []) if isinstance(raw_response, dict) else []
    normalized_results = []

    for raw_item in raw_results:
        if not isinstance(raw_item, dict):
            continue

        # A URL is required because source grounding depends on links. If a
        # result has no URL, the final answer cannot cite it usefully.
        url = str(raw_item.get("url", "")).strip()
        if not url:
            continue

        normalized_results.append(
            {
                "title": str(raw_item.get("title", "")).strip() or "Untitled",
                "url": url,
                "content": str(raw_item.get("content", "")).strip(),
                "score": raw_item.get("score", 0),
            }
        )

    return normalized_results
