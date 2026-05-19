"""Research agent orchestration.

This file is the "brain" of the app. It does not directly handle terminal
input, Docker, or `.env` files. Instead, it focuses on the agent workflow:

1. Turn the user's broad question into focused search queries.
2. Search the web for sources.
3. Ask the model whether the sources are enough.
4. Search again with follow-up queries when needed.
5. Ask the model to write a final answer using only the collected sources.

This is called orchestration: one piece of code coordinates several smaller
steps so they work together as a complete agent.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI

from config import ConfigError, Settings, debug_print, load_settings
from prompts import (
    ANSWER_PROMPT,
    FOLLOW_UP_QUERY_PROMPT,
    QUERY_PROMPT,
    SUFFICIENCY_PROMPT,
)
from tools import SearchError, search_web


# A "search function" is anything that accepts a query and returns a list of
# normalized search-result dictionaries. This type alias helps explain that the
# agent does not care whether the function talks to Tavily, a fake test object,
# or another search service later.
SearchFunction = Callable[..., list[dict[str, Any]]]


class ResearchAgent:
    """Coordinate a simple source-grounded research workflow.

    What this class does:
    - Uses OpenAI to create search queries and write answers.
    - Uses Tavily search, through `tools.search_web`, to collect web sources.
    - Repeats the search process when the current sources are not enough.

    Why it exists:
    - An "agent" is not magic. In this project, an agent is ordinary Python code
      that decides which step to run next: ask the model, call a tool, inspect
      the result, and continue.

    Inputs:
    - `settings`: API keys, model name, debug mode, and max iterations.
    - `openai_client`: object used to make OpenAI API calls.
    - `search_function`: function used to search the web.

    Outputs:
    - The class itself does not output anything when constructed.
    - Calling `answer(question)` returns a final answer string.

    Dependency injection note:
    - `openai_client` and `search_function` can be passed in from tests. That is
      called dependency injection. It lets tests use fake clients instead of
      making real network calls or spending API credits.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        openai_client: Any | None = None,
        search_function: SearchFunction = search_web,
    ) -> None:
        """Create a research agent.

        What this function does:
        - Stores configuration.
        - Creates the OpenAI client if one was not provided.
        - Stores the search function the agent should use.

        Why it exists:
        - Setup is separate from answering so one agent can answer multiple
          questions in the interactive terminal loop.

        Inputs:
        - `settings`: optional app configuration. If missing, read from `.env`.
        - `openai_client`: optional real or fake OpenAI client.
        - `search_function`: optional real or fake web search function.

        Outputs:
        - `None`. The initialized object is stored in `self`.
        """

        # If tests pass a Settings object, use it. Otherwise load real settings
        # from environment variables. This keeps production code and tests using
        # the same agent class.
        self.settings = settings or load_settings()

        # The OpenAI client is the object from the OpenAI Python SDK that knows
        # how to make HTTP requests to OpenAI's API. We keep it on `self` so all
        # methods can reuse the same client.
        self.openai_client = openai_client or OpenAI(api_key=self.settings.openai_api_key)

        # The search function is injected for testability. In real use it is
        # `search_web`, which calls Tavily. In tests it is a small fake function
        # that returns predictable sample data.
        self.search_function = search_function

    def answer(self, question: str) -> str:
        """Return a final research answer for one user question.

        What this function does:
        - Validates the question.
        - Runs the whole agent workflow.
        - Returns either an answer or a friendly error message.

        Why it exists:
        - This is the public method the CLI calls. Keeping one clear entry point
          makes the rest of the app easier to understand.

        Inputs:
        - `question`: the user's research question as plain text.

        Outputs:
        - A string containing the final answer, source-grounded failure message,
          or friendly error.

        Example flow:
        - User asks: "How do Python research agents work?"
        - Agent searches, checks sources, maybe searches again, then returns an
          answer with links.
        """

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Please provide a research question.")

        try:
            debug_print("Received question", cleaned_question, self.settings.debug)

            initial_search_queries = self.generate_search_queries(cleaned_question)
            debug_print(
                "Generated search queries",
                initial_search_queries,
                self.settings.debug,
            )

            collected_sources = self.run_research_loop(
                cleaned_question,
                initial_search_queries,
            )
            debug_print(
                "Collected unique search result count",
                len(collected_sources),
                self.settings.debug,
            )

            # Source grounding means the model should answer from collected
            # sources rather than from memory alone. If there are no sources, we
            # stop instead of asking the model to guess.
            if not collected_sources:
                return (
                    "I could not find useful search results for that question. "
                    "Try rephrasing it or checking your Tavily API key."
                )

            debug_print(
                "Synthesizing final answer from search results",
                enabled=self.settings.debug,
            )
            return self.synthesize_answer(cleaned_question, collected_sources)
        except (ConfigError, SearchError) as exc:
            # These are expected operational problems: missing keys or failed
            # search calls. We turn them into clean user-facing text.
            return f"Research agent error: {exc}"
        except Exception as exc:
            # A broad fallback keeps the CLI from crashing during the MVP. In a
            # larger project, we would log this error for maintainers too.
            return f"Research agent error: {exc}"

    def generate_search_queries(self, question: str) -> list[str]:
        """Ask OpenAI to turn one broad question into 2-3 search queries.

        What this function does:
        - Fills in `QUERY_PROMPT` with the user's question.
        - Sends that prompt to OpenAI.
        - Parses the model response as JSON.

        Why it exists:
        - Search engines work better with focused queries than vague questions.
          The LLM helps translate human wording into web-search-friendly text.

        Inputs:
        - `question`: the user's cleaned research question.

        Outputs:
        - A list of up to three query strings.

        Example flow:
        - Input question: "What is retrieval augmented generation?"
        - Output queries: ["retrieval augmented generation overview",
          "RAG AI explanation", "retrieval augmented generation examples"]
        """

        # Prompt engineering means carefully writing instructions for the model.
        # Here we explicitly ask for JSON so Python can parse the response.
        prompt = QUERY_PROMPT.format(question=question)

        debug_print("Calling OpenAI to generate search queries", enabled=self.settings.debug)
        response = self.openai_client.responses.create(
            model=self.settings.openai_model,
            input=prompt,
        )

        # `output_text` is the convenient text output provided by the OpenAI
        # Responses API. `.strip()` removes surrounding whitespace.
        model_text = getattr(response, "output_text", "").strip()
        search_queries = self._parse_query_json(model_text)

        # LLMs can occasionally return unexpected text. If JSON parsing fails,
        # we still have a useful fallback: search for the original question.
        if not search_queries:
            debug_print(
                "Query response was not valid JSON; using original question",
                model_text,
                self.settings.debug,
            )
            search_queries = [question]

        return search_queries[:3]

    def collect_search_results(self, queries: list[str]) -> list[dict[str, Any]]:
        """Search once and return the top unique sources.

        What this function does:
        - Runs one batch of web searches.
        - Deduplicates sources by URL.
        - Keeps the top eight results.

        Why it exists:
        - This helper is useful for tests and for understanding the simpler
          one-pass version of the research flow.

        Inputs:
        - `queries`: search strings to send to Tavily.

        Outputs:
        - A list of normalized source dictionaries.
        """

        return self.collect_new_search_results(queries, already_seen_urls=set())[:8]

    def collect_new_search_results(
        self,
        queries: list[str],
        already_seen_urls: set[str],
    ) -> list[dict[str, Any]]:
        """Search the web and return only sources we have not seen before.

        What this function does:
        - Calls the search tool for each query.
        - Skips duplicate URLs.
        - Sorts new results by search score.

        Why it exists:
        - The iterative loop can run multiple searches. Without deduplication,
          the same source may appear again and again, wasting context space and
          making the final answer less useful.

        Inputs:
        - `queries`: the search queries for this iteration.
        - `already_seen_urls`: a set that remembers URLs from earlier searches.

        Outputs:
        - New source dictionaries that were not already seen.
        """

        new_sources: list[dict[str, Any]] = []

        for search_query in queries:
            debug_print("Searching Tavily", search_query, self.settings.debug)

            # This is the tool-use part of the agent. The LLM does not search
            # the web by itself; Python calls Tavily and gives the model the
            # source snippets later.
            search_results = self.search_function(
                search_query,
                api_key=self.settings.tavily_api_key,
                max_results=5,
            )
            debug_print(
                "Tavily result count",
                {"query": search_query, "result_count": len(search_results)},
                self.settings.debug,
            )

            for source in search_results:
                source_url = source["url"]

                # Deduplication uses a set because checking membership in a set
                # is fast and the intent is clear: "Have we seen this URL?"
                if source_url in already_seen_urls:
                    debug_print("Skipping duplicate source", source_url, self.settings.debug)
                    continue

                already_seen_urls.add(source_url)
                new_sources.append(source)

        # Higher Tavily scores should be more relevant. We sort descending so
        # the best-looking sources come first.
        new_sources.sort(key=lambda source: source.get("score") or 0, reverse=True)
        return new_sources

    def run_research_loop(
        self,
        question: str,
        initial_queries: list[str],
    ) -> list[dict[str, Any]]:
        """Run iterative search until sources are sufficient or time runs out.

        What this function does:
        - Searches with the current queries.
        - Adds new unique sources to the source list.
        - Asks OpenAI whether the sources are good enough.
        - Generates follow-up queries if more research is needed.
        - Stops after `settings.max_iterations`.

        Why it exists:
        - Real research is often iterative. You search, inspect what you found,
          notice what is missing, and search again with better queries. This
          method teaches that loop in simple Python.

        Inputs:
        - `question`: the user's original question.
        - `initial_queries`: first queries generated from the question.

        Outputs:
        - A deduplicated list of up to eight source dictionaries.
        """

        collected_sources: list[dict[str, Any]] = []
        already_seen_urls: set[str] = set()
        current_queries = initial_queries

        for iteration_number in range(1, self.settings.max_iterations + 1):
            debug_print(
                "Research iteration",
                {
                    "iteration": iteration_number,
                    "max_iterations": self.settings.max_iterations,
                    "queries": current_queries,
                },
                self.settings.debug,
            )

            new_sources = self.collect_new_search_results(
                current_queries,
                already_seen_urls,
            )
            collected_sources.extend(new_sources)

            # Keep only the best few sources. This is a beginner-friendly way to
            # avoid sending too much text into later prompts.
            collected_sources.sort(
                key=lambda source: source.get("score") or 0,
                reverse=True,
            )
            collected_sources = collected_sources[:8]

            debug_print(
                "Research iteration result count",
                {
                    "iteration": iteration_number,
                    "new_sources": len(new_sources),
                    "total_sources": len(collected_sources),
                },
                self.settings.debug,
            )

            # If the first searches found nothing, there is no source context to
            # evaluate or use for follow-up query generation.
            if not collected_sources:
                debug_print("No sources collected yet", enabled=self.settings.debug)
                break

            sufficiency_decision = self.check_source_sufficiency(
                question,
                collected_sources,
            )
            debug_print(
                "Source sufficiency decision",
                sufficiency_decision,
                self.settings.debug,
            )

            if sufficiency_decision["sufficient"]:
                break

            if iteration_number == self.settings.max_iterations:
                break

            current_queries = self.generate_follow_up_queries(
                question,
                collected_sources,
                sufficiency_decision["reason"],
            )
            debug_print("Generated follow-up queries", current_queries, self.settings.debug)

            # If the model cannot suggest follow-up queries, continuing the loop
            # would repeat work. Stopping here keeps the behavior predictable.
            if not current_queries:
                break

        return collected_sources

    def check_source_sufficiency(
        self,
        question: str,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask OpenAI whether the current sources are enough.

        What this function does:
        - Formats the current source list.
        - Sends a review prompt to OpenAI.
        - Parses the model's JSON decision.

        Why it exists:
        - The agent needs a simple stopping rule. Instead of always doing three
          searches, it asks whether the collected sources are sufficient.

        Inputs:
        - `question`: the user's original question.
        - `results`: current source dictionaries.

        Outputs:
        - A dictionary like:
          `{"sufficient": True, "reason": "Sources cover the question."}`
        """

        prompt = SUFFICIENCY_PROMPT.format(
            question=question,
            sources=self._format_sources(results),
        )

        debug_print("Calling OpenAI to check source sufficiency", enabled=self.settings.debug)
        response = self.openai_client.responses.create(
            model=self.settings.openai_model,
            input=prompt,
        )

        model_text = getattr(response, "output_text", "").strip()
        sufficiency_decision = self._parse_sufficiency_json(model_text)
        if sufficiency_decision is None:
            debug_print(
                "Sufficiency response was not valid JSON; continuing research",
                model_text,
                self.settings.debug,
            )
            return {"sufficient": False, "reason": "The sufficiency check was unclear."}

        return sufficiency_decision

    def generate_follow_up_queries(
        self,
        question: str,
        results: list[dict[str, Any]],
        reason: str,
    ) -> list[str]:
        """Ask OpenAI for follow-up queries when sources are not enough.

        What this function does:
        - Shows the model the original question, current sources, and what is
          missing.
        - Asks for one or two more focused search queries.

        Why it exists:
        - Follow-up queries make the loop adaptive. The second search can target
          gaps found in the first search instead of blindly repeating it.

        Inputs:
        - `question`: the original user question.
        - `results`: sources collected so far.
        - `reason`: why the current sources are insufficient.

        Outputs:
        - A list of up to two query strings.
        """

        prompt = FOLLOW_UP_QUERY_PROMPT.format(
            question=question,
            sources=self._format_sources(results),
            reason=reason,
        )

        debug_print("Calling OpenAI to generate follow-up queries", enabled=self.settings.debug)
        response = self.openai_client.responses.create(
            model=self.settings.openai_model,
            input=prompt,
        )

        model_text = getattr(response, "output_text", "").strip()
        follow_up_queries = self._parse_query_json(model_text)
        return follow_up_queries[:2]

    def synthesize_answer(
        self,
        question: str,
        results: list[dict[str, Any]],
    ) -> str:
        """Ask OpenAI to write the final source-grounded answer.

        What this function does:
        - Formats the source list.
        - Sends the answer prompt to OpenAI.
        - Returns the model's final text.

        Why it exists:
        - Search results are snippets, not a polished answer. The LLM is useful
          for combining the snippets into a readable explanation with links.

        Inputs:
        - `question`: the original user question.
        - `results`: final source list.

        Outputs:
        - A final answer string.
        """

        prompt = ANSWER_PROMPT.format(
            question=question,
            sources=self._format_sources(results),
        )

        debug_print("Calling OpenAI to synthesize the final answer", enabled=self.settings.debug)
        response = self.openai_client.responses.create(
            model=self.settings.openai_model,
            input=prompt,
        )

        final_answer = getattr(response, "output_text", "").strip()
        if not final_answer:
            return "The model did not return an answer. Please try again."

        return final_answer

    def _parse_query_json(self, text: str) -> list[str]:
        """Parse a model response that should contain a JSON list of queries.

        What this function does:
        - Attempts to convert text into Python data with `json.loads`.
        - Verifies that the data is a list.
        - Keeps only non-empty strings.

        Why it exists:
        - LLM responses are text. When we ask the model for JSON, Python still
          has to parse and validate that text before trusting it.

        Inputs:
        - `text`: raw model output.

        Outputs:
        - A list of clean query strings. Returns an empty list if parsing fails.
        """

        try:
            parsed_json = json.loads(text)
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed_json, list):
            return []

        return [
            item.strip()
            for item in parsed_json
            if isinstance(item, str) and item.strip()
        ]

    def _parse_sufficiency_json(self, text: str) -> dict[str, Any] | None:
        """Parse a model response that should contain a sufficiency decision.

        What this function does:
        - Attempts to parse JSON.
        - Verifies that the result is a dictionary.
        - Verifies that `sufficient` is really a boolean.
        - Provides a fallback reason when the model omits one.

        Why it exists:
        - The agent's loop depends on this decision. Badly shaped model output
          should not crash the app or accidentally stop research too early.

        Inputs:
        - `text`: raw model output.

        Outputs:
        - A dictionary with `sufficient` and `reason`, or `None` if parsing
          fails.
        """

        try:
            parsed_json = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed_json, dict):
            return None

        sufficient = parsed_json.get("sufficient")
        reason = str(parsed_json.get("reason", "")).strip()
        if not isinstance(sufficient, bool):
            return None

        return {
            "sufficient": sufficient,
            "reason": reason or "No reason provided.",
        }

    def _format_sources(self, results: list[dict[str, Any]]) -> str:
        """Turn source dictionaries into text that can be placed in a prompt.

        What this function does:
        - Converts each source into a readable block with title, URL, and
          snippet.
        - Joins those blocks together for the prompt.

        Why it exists:
        - Models do not understand Python dictionaries directly in a reliable
          way. Formatting sources as clear text makes the prompt easier for the
          model to follow.

        Inputs:
        - `results`: normalized source dictionaries.

        Outputs:
        - A string containing numbered source notes.
        """

        formatted_source_blocks = []
        for index, source in enumerate(results, start=1):
            formatted_source_blocks.append(
                "\n".join(
                    [
                        f"Source {index}: {source['title']}",
                        f"URL: {source['url']}",
                        f"Snippet: {source.get('content', '')}",
                    ]
                )
            )

        return "\n\n".join(formatted_source_blocks)
