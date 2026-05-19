"""Prompt templates used by the research agent.

Keeping prompts here makes the agent flow easier to read and tweak.
"""

QUERY_PROMPT = """You help create web search queries for beginner research.

Given the user's research question, generate 2 or 3 concise search queries.
Return only a JSON array of strings. Do not include markdown.

Question:
{question}
"""


ANSWER_PROMPT = """You are a careful research assistant.

Answer the user's question using only the provided web search results.
Include source links in the answer. If the results are not enough to answer
confidently, say what is missing.

User question:
{question}

Search results:
{sources}
"""
