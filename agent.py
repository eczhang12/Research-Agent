"""Research agent orchestration.

The ResearchAgent turns a question into search queries, retrieves sources, and
asks OpenAI to write a source-grounded final answer.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI

from config import ConfigError, Settings, debug_print, load_settings
from prompts import ANSWER_PROMPT, QUERY_PROMPT
from tools import SearchError, search_web


SearchFunction = Callable[..., list[dict[str, Any]]]


class ResearchAgent:
    """Small beginner-friendly research agent."""

    def __init__(
        self,
        settings: Settings | None = None,
        openai_client: Any | None = None,
        search_function: SearchFunction = search_web,
    ) -> None:
        self.settings = settings or load_settings()
        self.openai_client = openai_client or OpenAI(api_key=self.settings.openai_api_key)
        self.search_function = search_function

    def answer(self, question: str) -> str:
        """Return an answer with sources for a user research question."""

        question = question.strip()
        if not question:
            raise ValueError("Please provide a research question.")

        try:
            debug_print("Received question", question, self.settings.debug)
            queries = self.generate_search_queries(question)
            debug_print("Generated search queries", queries, self.settings.debug)
            results = self.collect_search_results(queries)
            debug_print("Collected unique search result count", len(results), self.settings.debug)
            if not results:
                return (
                    "I could not find useful search results for that question. "
                    "Try rephrasing it or checking your Tavily API key."
                )

            debug_print("Synthesizing final answer from search results", enabled=self.settings.debug)
            return self.synthesize_answer(question, results)
        except (ConfigError, SearchError) as exc:
            return f"Research agent error: {exc}"
        except Exception as exc:
            return f"Research agent error: {exc}"

    def generate_search_queries(self, question: str) -> list[str]:
        """Ask OpenAI for 2-3 useful search queries."""

        prompt = QUERY_PROMPT.format(question=question)
        debug_print("Calling OpenAI to generate search queries", enabled=self.settings.debug)
        response = self.openai_client.responses.create(
            model=self.settings.openai_model,
            input=prompt,
        )

        text = getattr(response, "output_text", "").strip()
        queries = self._parse_query_json(text)

        if not queries:
            debug_print(
                "Query response was not valid JSON; using original question",
                text,
                self.settings.debug,
            )
            queries = [question]

        return queries[:3]

    def collect_search_results(self, queries: list[str]) -> list[dict[str, Any]]:
        """Run searches, deduplicate by URL, and keep the best few sources."""

        seen_urls = set()
        results: list[dict[str, Any]] = []

        for query in queries:
            debug_print("Searching Tavily", query, self.settings.debug)
            search_results = self.search_function(
                query,
                api_key=self.settings.tavily_api_key,
                max_results=5,
            )
            debug_print(
                "Tavily result count",
                {"query": query, "result_count": len(search_results)},
                self.settings.debug,
            )

            for result in search_results:
                url = result["url"]
                if url in seen_urls:
                    debug_print("Skipping duplicate source", url, self.settings.debug)
                    continue
                seen_urls.add(url)
                results.append(result)

        results.sort(key=lambda item: item.get("score") or 0, reverse=True)
        return results[:8]

    def synthesize_answer(
        self,
        question: str,
        results: list[dict[str, Any]],
    ) -> str:
        """Ask OpenAI to produce the final answer from normalized sources."""

        prompt = ANSWER_PROMPT.format(
            question=question,
            sources=self._format_sources(results),
        )
        debug_print("Calling OpenAI to synthesize the final answer", enabled=self.settings.debug)
        response = self.openai_client.responses.create(
            model=self.settings.openai_model,
            input=prompt,
        )

        answer = getattr(response, "output_text", "").strip()
        if not answer:
            return "The model did not return an answer. Please try again."

        return answer

    def _parse_query_json(self, text: str) -> list[str]:
        """Parse the model's JSON query list with a simple safe fallback."""

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        return [item.strip() for item in data if isinstance(item, str) and item.strip()]

    def _format_sources(self, results: list[dict[str, Any]]) -> str:
        """Turn search results into compact source notes for the answer prompt."""

        lines = []
        for index, result in enumerate(results, start=1):
            lines.append(
                "\n".join(
                    [
                        f"Source {index}: {result['title']}",
                        f"URL: {result['url']}",
                        f"Snippet: {result.get('content', '')}",
                    ]
                )
            )

        return "\n\n".join(lines)
