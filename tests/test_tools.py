import pytest

from tools import SearchError, search_web


class FakeTavilyClient:
    def __init__(self, response=None, error=None):
        self.response = response or {}
        self.error = error
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_search_web_normalizes_tavily_results():
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
    client = FakeTavilyClient({"results": []})

    assert search_web("nothing obscure", "fake-key", client=client) == []


def test_search_web_wraps_api_failures():
    client = FakeTavilyClient(error=RuntimeError("network down"))

    with pytest.raises(SearchError, match="Search failed"):
        search_web("latest research", "fake-key", client=client)
