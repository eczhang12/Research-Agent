"""Command-line entry point for the research agent.

This file handles the terminal experience:

- Print a welcome message.
- Read user questions.
- Let the user ask multiple questions in one session.
- Exit cleanly when the user types `exit`, `quit`, or `q`.

The important architecture rule is that `main.py` does not perform research
itself. It delegates all research work to `ResearchAgent` in `agent.py`.
"""

import sys

from agent import ResearchAgent
from config import ConfigError


# These commands are intentionally short and common in CLIs. The set makes exit
# detection easy to read: `some_text in EXIT_COMMANDS`.
EXIT_COMMANDS = {"exit", "quit", "q"}


def is_exit_command(user_input: str) -> bool:
    """Check whether terminal input means "leave the program".

    What this function does:
    - Trims spaces.
    - Ignores capitalization.
    - Compares the input to the known exit commands.

    Why it exists:
    - Keeping this logic in a small function makes the loop easier to read and
      easy to test without starting the full app.

    Inputs:
    - `user_input`: raw text typed by the user.

    Outputs:
    - `True` if the user wants to exit, otherwise `False`.
    """

    return user_input.strip().lower() in EXIT_COMMANDS


def answer_question(agent: ResearchAgent, question: str) -> bool:
    """Answer one terminal question using the research agent.

    What this function does:
    - Rejects empty input with a friendly message.
    - Calls `agent.answer(question)` for real questions.
    - Prints the returned answer.

    Why it exists:
    - The terminal loop should only coordinate input/output. This helper keeps
      "handle one question" separate from "keep asking for more questions".

    Inputs:
    - `agent`: a `ResearchAgent` instance.
    - `question`: raw question text from the terminal or command-line argument.

    Outputs:
    - `True` when a real question was handled.
    - `False` when the input was empty.
    """

    cleaned_question = question.strip()
    if not cleaned_question:
        print("Please enter a real research question.")
        return False

    # Blank lines around the answer make the terminal output easier to scan.
    print()
    print(agent.answer(cleaned_question))
    print()
    return True


def run_interactive_loop(agent: ResearchAgent) -> int:
    """Run the multi-question terminal session.

    What this function does:
    - Prints a welcome message.
    - Repeatedly asks for a question.
    - Exits when the user types an exit command.
    - Sends each real question to the agent.

    Why it exists:
    - A loop lets one agent instance answer many questions without restarting
      Docker each time.

    Inputs:
    - `agent`: a configured `ResearchAgent`.

    Outputs:
    - Process-style exit code `0` for a clean exit.
    """

    print("Welcome to the Research Agent MVP.")
    print("Ask a research question, or type exit, quit, or q to leave.")
    print()

    while True:
        user_question = input("Research question: ")
        if is_exit_command(user_question):
            print("Goodbye.")
            return 0

        answer_question(agent, user_question)


def main() -> int:
    """Create the agent and run either one-shot or interactive CLI mode.

    What this function does:
    - Builds a `ResearchAgent`.
    - If a question was passed as command-line arguments, answers it once.
    - Otherwise starts the interactive loop.
    - Handles setup errors and Ctrl+C cleanly.

    Why it exists:
    - Python calls this function from the `if __name__ == "__main__"` block.
      Returning exit codes makes CLI behavior predictable.

    Inputs:
    - None directly. It reads `sys.argv`, which contains command-line args.

    Outputs:
    - Integer exit code:
      - `0` means success or graceful user exit.
      - `1` means configuration/setup failed.
    """

    try:
        # Creating the agent loads configuration and prepares API clients.
        agent = ResearchAgent()

        # This one-shot mode is handy for scripts and tests:
        # `python main.py "What is RAG?"`
        if len(sys.argv) > 1:
            answer_question(agent, " ".join(sys.argv[1:]))
            return 0

        return run_interactive_loop(agent)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    except KeyboardInterrupt:
        # Ctrl+C raises KeyboardInterrupt. Catching it prevents an unfriendly
        # stack trace and gives the user a clean goodbye.
        print("\nGoodbye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
