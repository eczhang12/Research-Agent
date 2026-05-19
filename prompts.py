"""Prompt templates used by the research agent.

Prompts are instructions we send to the language model. Prompt engineering is
the practice of writing those instructions clearly enough that the model returns
useful, predictable output.

This file keeps prompts separate from Python logic so beginners can see the two
parts of an AI agent:

- Python controls the workflow.
- Prompts tell the model what each AI step should do.
"""

# We ask the LLM to turn the user's broad research question into smaller
# web-search-friendly queries because search engines work better with focused
# questions than with vague requests. We request JSON so Python can parse the
# result into a list.
QUERY_PROMPT = """You help create web search queries for beginner research.

Given the user's research question, generate 2 or 3 concise search queries.
Return only a JSON array of strings. Do not include markdown.

Question:
{question}
"""


# Follow-up queries are used after the first search results are judged
# insufficient. Giving the model the current sources and the reason they are not
# enough helps it create better next searches.
FOLLOW_UP_QUERY_PROMPT = """You help improve web research when the first search was not enough.

Given the user's question, the sources already collected, and the reason more
research is needed, generate 1 or 2 new search queries that look for missing
information. Return only a JSON array of strings. Do not include markdown.

Question:
{question}

Known sources:
{sources}

Reason more research is needed:
{reason}
"""


# This prompt asks the model to make a simple control-flow decision for the
# agent: should we stop searching, or do we need another iteration? The strict
# JSON shape lets Python read the decision safely.
SUFFICIENCY_PROMPT = """You review web search results for a research agent.

Decide whether the collected sources are sufficient to answer the user's
question. Return only JSON in this shape:
{{"sufficient": true, "reason": "short reason"}}

Use false when the sources are too thin, irrelevant, contradictory, or missing
important context.

Question:
{question}

Search results:
{sources}
"""


# This is the final synthesis prompt. "Source-grounded" means the model should
# answer using the provided search results, not just its own memory. That helps
# the final answer include links and makes the reasoning easier to inspect.
ANSWER_PROMPT = """You are a careful research assistant.

Answer the user's question using only the provided web search results.
Include source links in the answer. If the results are not enough to answer
confidently, say what is missing.

User question:
{question}

Search results:
{sources}
"""
