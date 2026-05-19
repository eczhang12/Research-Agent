from types import SimpleNamespace

import pytest

from agent import ResearchAgent
from config import Settings


class FakeResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.outputs.pop(0))


class FakeOpenAIClient:
    def __init__(self, outputs):
        self.responses = FakeResponses(outputs)


def make_agent(openai_outputs, search_results):
    settings = Settings(
        openai_api_key="fake-openai-key",
        tavily_api_key="fake-tavily-key",
        openai_model="test-model",
    )

    def fake_search(query, api_key, max_results):
        return search_results.get(query, [])

    return ResearchAgent(
        settings=settings,
        openai_client=FakeOpenAIClient(openai_outputs),
        search_function=fake_search,
    )


def test_answer_rejects_empty_question():
    agent = make_agent([], {})

    with pytest.raises(ValueError, match="research question"):
        agent.answer("   ")


def test_answer_includes_sources_from_mocked_results():
    agent = make_agent(
        [
            '["python research agent tutorial", "web search agent python"]',
            "Python research agents combine search and synthesis.\n\nSources: https://example.com",
        ],
        {
            "python research agent tutorial": [
                {
                    "title": "Tutorial",
                    "url": "https://example.com",
                    "content": "Python agent tutorial",
                    "score": 0.9,
                }
            ],
            "web search agent python": [
                {
                    "title": "Duplicate",
                    "url": "https://example.com",
                    "content": "Duplicate URL",
                    "score": 0.5,
                }
            ],
        },
    )

    answer = agent.answer("How do Python research agents work?")

    assert "https://example.com" in answer
    assert len(agent.openai_client.responses.calls) == 2


def test_answer_handles_no_search_results():
    agent = make_agent(['["no match query"]'], {"no match query": []})

    answer = agent.answer("What is a deliberately obscure thing?")

    assert "could not find useful search results" in answer


def test_generate_search_queries_falls_back_to_question_when_json_is_invalid():
    agent = make_agent(["not json"], {})

    assert agent.generate_search_queries("What is retrieval augmented generation?") == [
        "What is retrieval augmented generation?"
    ]


def test_debug_mode_prints_labeled_progress_without_changing_answer(capsys):
    settings = Settings(
        openai_api_key="fake-openai-key",
        tavily_api_key="fake-tavily-key",
        openai_model="test-model",
        debug=True,
    )

    def fake_search(query, api_key, max_results):
        return [
            {
                "title": "Source",
                "url": "https://example.com",
                "content": "Useful snippet",
                "score": 0.8,
            }
        ]

    agent = ResearchAgent(
        settings=settings,
        openai_client=FakeOpenAIClient(
            [
                '["debug search query"]',
                "Final answer with source: https://example.com",
            ]
        ),
        search_function=fake_search,
    )

    answer = agent.answer("Debug this question")

    output = capsys.readouterr().out
    assert answer == "Final answer with source: https://example.com"
    assert "[debug] Received question" in output
    assert "Debug this question" in output
    assert "[debug] Generated search queries" in output
    assert '"debug search query"' in output
