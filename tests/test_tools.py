"""Tests for the Tavily search wrapper.

These tests focus on `tools.search_web`, not the whole agent. They use a fake
Tavily client so the tests can run inside Docker without internet access or API
keys.
"""

import pytest

from tools import SearchError, search_web


class FakeTavilyClient:
    """A fake Tavily client with the same `.search()` method the app uses."""

    def __init__(self, response=None, error=None):
        """Store either a fake API response or a fake error to raise."""

        self.response = response or {}
        self.error = error
        self.calls = []

    def search(self, **kwargs):
        """Pretend to search Tavily and record how it was called."""

        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_search_web_normalizes_tavily_results():
    """Raw Tavily-style results should become simple source dictionaries."""

    client = FakeTavilyClient(
        {
            "results": [
                {
                    "title": "Example Result",
                    "url": "https://example.com",
                    "content": "A useful snippet",
                    "score": 0.91,
                },
                {"title": "Missing URL"},
            ]
        }
    )

    results = search_web("python research agents", "fake-key", client=client)

    assert client.calls == [{"query": "python research agents", "max_results": 5}]
    assert results == [
        {
            "title": "Example Result",
            "url": "https://example.com",
            "content": "A useful snippet",
            "score": 0.91,
        }
    ]


def test_search_web_returns_empty_list_for_empty_results():
    """No Tavily results should become an empty Python list."""

    client = FakeTavilyClient({"results": []})

    assert search_web("nothing obscure", "fake-key", client=client) == []


def test_search_web_wraps_api_failures():
    """Provider errors should be wrapped in our beginner-friendly SearchError."""

    client = FakeTavilyClient(error=RuntimeError("network down"))

    with pytest.raises(SearchError, match="Search failed"):
        search_web("latest research", "fake-key", client=client)
