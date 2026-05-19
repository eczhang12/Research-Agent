"""Tests for the command-line helper functions.

The interactive loop normally waits for real keyboard input. Tests replace
`input()` with fake values so we can verify the loop without a human typing.
"""

import builtins

import pytest

from main import answer_question, is_exit_command, run_interactive_loop


class FakeAgent:
    """A tiny fake agent that records questions and returns simple answers."""

    def __init__(self):
        """Prepare a list so tests can inspect which questions were asked."""

        self.questions = []

    def answer(self, question):
        """Pretend to answer a question without doing real research."""

        self.questions.append(question)
        return f"answer for {question}"


@pytest.mark.parametrize("command", ["exit", "quit", "q", " EXIT ", "Quit"])
def test_is_exit_command_detects_supported_commands(command):
    """The CLI should recognize all supported exit commands."""

    assert is_exit_command(command)


@pytest.mark.parametrize("text", ["", "research question", "queue"])
def test_is_exit_command_rejects_other_input(text):
    """Normal questions should not be mistaken for exit commands."""

    assert not is_exit_command(text)


def test_answer_question_handles_empty_input(capsys):
    """Empty input should print guidance and avoid calling the agent."""

    agent = FakeAgent()

    handled = answer_question(agent, "   ")

    assert handled is False
    assert agent.questions == []
    assert "Please enter a real research question." in capsys.readouterr().out


def test_answer_question_calls_agent_for_real_question(capsys):
    """Real input should be passed to the agent and printed."""

    agent = FakeAgent()

    handled = answer_question(agent, "What is AI safety?")

    assert handled is True
    assert agent.questions == ["What is AI safety?"]
    assert "answer for What is AI safety?" in capsys.readouterr().out


def test_run_interactive_loop_handles_empty_then_question_then_exit(
    monkeypatch,
    capsys,
):
    """The loop should handle empty input, a real question, and exit."""

    agent = FakeAgent()
    inputs = iter(["", "What is RAG?", "q"])

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    exit_code = run_interactive_loop(agent)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert agent.questions == ["What is RAG?"]
    assert "Welcome to the Research Agent MVP." in output
    assert "Please enter a real research question." in output
    assert "Goodbye." in output
