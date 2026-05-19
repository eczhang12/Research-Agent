"""Tests for the research-agent orchestration code.

These tests intentionally use fake OpenAI and fake search objects. That teaches
an important testing idea: we can test our Python workflow without calling real
APIs, spending money, or depending on the internet.
"""

from types import SimpleNamespace

import pytest

from agent import ResearchAgent
from config import Settings


class FakeResponses:
    """A tiny fake version of `openai_client.responses`.

    What this class does:
    - Stores pre-written model outputs.
    - Records every call the agent makes.

    Why it exists:
    - Tests need predictable model responses. A real model may answer
      differently each time, but this fake returns exactly what the test asks
      for.
    """

    def __init__(self, outputs):
        """Store fake response texts and prepare a call log."""

        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        """Pretend to call OpenAI and return the next fake response.

        Inputs:
        - `kwargs`: the same keyword arguments the real OpenAI SDK would get.

        Outputs:
        - A small object with an `output_text` attribute, matching what the
          agent reads from the real SDK.
        """

        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.outputs.pop(0))


class FakeOpenAIClient:
    """A fake top-level OpenAI client used by `ResearchAgent` tests."""

    def __init__(self, outputs):
        """Attach a fake `.responses` object to match the real SDK shape."""

        self.responses = FakeResponses(outputs)


def make_agent(openai_outputs, search_results, max_iterations=3):
    """Build a `ResearchAgent` with fake dependencies.

    What this function does:
    - Creates fake settings.
    - Creates a fake search function.
    - Creates a fake OpenAI client.
    - Returns a ready-to-test agent.

    Why it exists:
    - Many tests need the same setup. This helper keeps each test focused on
      the behavior it is checking.

    Inputs:
    - `openai_outputs`: model responses the fake client should return.
    - `search_results`: mapping of query text to fake Tavily results.
    - `max_iterations`: research-loop limit for the test.

    Outputs:
    - A `ResearchAgent` that does not call real external services.
    """

    settings = Settings(
        openai_api_key="fake-openai-key",
        tavily_api_key="fake-tavily-key",
        openai_model="test-model",
        max_iterations=max_iterations,
    )

    def fake_search(query, api_key, max_results):
        """Return fake search results for one query."""

        return search_results.get(query, [])

    return ResearchAgent(
        settings=settings,
        openai_client=FakeOpenAIClient(openai_outputs),
        search_function=fake_search,
    )


def test_answer_rejects_empty_question():
    """Empty user input should fail before any API-like work happens."""

    agent = make_agent([], {})

    with pytest.raises(ValueError, match="research question"):
        agent.answer("   ")


def test_answer_includes_sources_from_mocked_results():
    """A normal answer should include source links from mocked search results."""

    agent = make_agent(
        [
            '["python research agent tutorial", "web search agent python"]',
            '{"sufficient": true, "reason": "Enough sources."}',
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
    assert len(agent.openai_client.responses.calls) == 3


def test_answer_handles_no_search_results():
    """When Tavily returns no results, the agent should not ask the LLM to guess."""

    agent = make_agent(['["no match query"]'], {"no match query": []})

    answer = agent.answer("What is a deliberately obscure thing?")

    assert "could not find useful search results" in answer


def test_generate_search_queries_falls_back_to_question_when_json_is_invalid():
    """Invalid model JSON should fall back to searching the original question."""

    agent = make_agent(["not json"], {})

    assert agent.generate_search_queries("What is retrieval augmented generation?") == [
        "What is retrieval augmented generation?"
    ]


def test_debug_mode_prints_labeled_progress_without_changing_answer(capsys):
    """Debug mode should print internals but keep the final answer unchanged."""

    settings = Settings(
        openai_api_key="fake-openai-key",
        tavily_api_key="fake-tavily-key",
        openai_model="test-model",
        debug=True,
    )

    def fake_search(query, api_key, max_results):
        """Return one source so the debug-mode flow can complete."""

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
                '{"sufficient": true, "reason": "Enough sources."}',
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


def test_research_loop_generates_follow_up_queries_when_sources_are_insufficient():
    """The agent should search again when sources are judged insufficient."""

    search_calls = []
    settings = Settings(
        openai_api_key="fake-openai-key",
        tavily_api_key="fake-tavily-key",
        openai_model="test-model",
        max_iterations=3,
    )
    search_results = {
        "initial query": [
            {
                "title": "Thin source",
                "url": "https://example.com/thin",
                "content": "A partial answer",
                "score": 0.4,
            }
        ],
        "follow up query": [
            {
                "title": "Better source",
                "url": "https://example.com/better",
                "content": "A fuller answer",
                "score": 0.9,
            },
            {
                "title": "Duplicate thin source",
                "url": "https://example.com/thin",
                "content": "Duplicate URL",
                "score": 0.8,
            },
        ],
    }

    def fake_search(query, api_key, max_results):
        """Record each query and return the matching fake search results."""

        search_calls.append(query)
        return search_results.get(query, [])

    agent = ResearchAgent(
        settings=settings,
        openai_client=FakeOpenAIClient(
            [
                '["initial query"]',
                '{"sufficient": false, "reason": "Need a stronger source."}',
                '["follow up query"]',
                '{"sufficient": true, "reason": "The follow-up source is enough."}',
                "Final answer with source: https://example.com/better",
            ]
        ),
        search_function=fake_search,
    )

    answer = agent.answer("How does the loop work?")

    assert answer == "Final answer with source: https://example.com/better"
    assert search_calls == ["initial query", "follow up query"]
    assert len(agent.openai_client.responses.calls) == 5


def test_research_loop_stops_at_max_iterations():
    """The loop should synthesize an answer after reaching the iteration limit."""

    settings = Settings(
        openai_api_key="fake-openai-key",
        tavily_api_key="fake-tavily-key",
        openai_model="test-model",
        max_iterations=1,
    )

    def fake_search(query, api_key, max_results):
        """Return one incomplete source for the max-iteration test."""

        return [
            {
                "title": "Only source",
                "url": "https://example.com/only",
                "content": "Still incomplete",
                "score": 0.5,
            }
        ]

    agent = ResearchAgent(
        settings=settings,
        openai_client=FakeOpenAIClient(
            [
                '["initial query"]',
                '{"sufficient": false, "reason": "Need more."}',
                "Final answer after max iterations.",
            ]
        ),
        search_function=fake_search,
    )

    answer = agent.answer("Stop after one iteration?")

    assert answer == "Final answer after max iterations."
    assert len(agent.openai_client.responses.calls) == 3
