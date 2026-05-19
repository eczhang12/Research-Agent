"""Command-line entry point for the research agent.

Run this file through Docker Compose so all dependencies stay inside the
container.
"""

import sys

from agent import ResearchAgent
from config import ConfigError


EXIT_COMMANDS = {"exit", "quit", "q"}


def is_exit_command(user_input: str) -> bool:
    """Return True when the user wants to leave the interactive loop."""

    return user_input.strip().lower() in EXIT_COMMANDS


def answer_question(agent: ResearchAgent, question: str) -> bool:
    """Answer one question.

    Returns True when a real question was handled and False when the input was
    empty. Keeping this small helper makes the terminal loop easy to test.
    """

    question = question.strip()
    if not question:
        print("Please enter a real research question.")
        return False

    print()
    print(agent.answer(question))
    print()
    return True


def run_interactive_loop(agent: ResearchAgent) -> int:
    """Run the multi-question terminal session."""

    print("Welcome to the Research Agent MVP.")
    print("Ask a research question, or type exit, quit, or q to leave.")
    print()

    while True:
        question = input("Research question: ")
        if is_exit_command(question):
            print("Goodbye.")
            return 0

        answer_question(agent, question)


def main() -> int:
    """Run the CLI app and return a process exit code."""

    try:
        agent = ResearchAgent()
        if len(sys.argv) > 1:
            answer_question(agent, " ".join(sys.argv[1:]))
            return 0

        return run_interactive_loop(agent)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nGoodbye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
