"""Configuration helpers for the research agent.

This project gets settings from environment variables instead of hard-coding
values in Python files. That matters for two beginner-friendly reasons:

1. API keys are secrets. They should live in `.env`, not in source code.
2. Settings like debug mode or the model name should be easy to change without
   editing the agent logic.

Docker Compose loads `.env` into the container, and this file turns those raw
strings into a typed `Settings` object that the rest of the app can use.
"""

from dataclasses import dataclass
import json
import os

from dotenv import load_dotenv


# Default settings keep optional environment variables predictable. API keys are
# not optional, so `load_settings()` validates those separately.
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_ITERATIONS = 3


DEBUG_COLOR_LABEL = "\x1B[33m"
DEBUG_COLOR = "\033[36m"
RESET_COLOR = "\033[0m"


class ConfigError(Exception):
    """Explain configuration problems in a user-friendly way.

    What this class does:
    - Gives the app a specific exception type for missing/invalid settings.

    Why it exists:
    - Catching `ConfigError` in `main.py` lets us print a helpful setup message
      instead of showing a scary Python traceback to a beginner.

    Inputs:
    - A normal exception message string.

    Outputs:
    - No return value. Raising this exception stops normal execution until it
      is caught by error-handling code.
    """


@dataclass(frozen=True)
class Settings:
    """A small container for all settings the app needs at runtime.

    What this class does:
    - Stores API keys, model choices, debug mode, and iteration limits.

    Why it exists:
    - Passing one `Settings` object around is clearer than passing many
      unrelated strings and booleans into every function.
    - `frozen=True` means the object should not be changed after creation,
      which prevents accidental setting changes while the agent is running.

    Inputs:
    - `openai_api_key`: secret key used to call OpenAI.
    - `tavily_api_key`: secret key used to call Tavily search.
    - `openai_model`: model name used for all OpenAI calls.
    - `debug`: whether to print step-by-step debug output.
    - `max_iterations`: maximum number of research/search rounds.

    Outputs:
    - A `Settings` instance, such as:
      `Settings(openai_api_key="...", tavily_api_key="...", debug=False)`.
    """

    openai_api_key: str
    tavily_api_key: str
    openai_model: str = DEFAULT_MODEL
    debug: bool = False
    max_iterations: int = DEFAULT_MAX_ITERATIONS


def parse_bool(value: str | None) -> bool:
    """Convert an environment variable string into a Python boolean.

    What this function does:
    - Treats values like `"true"`, `"yes"`, `"on"`, and `"1"` as `True`.
    - Treats missing or other values as `False`.

    Why it exists:
    - Environment variables are always strings, but the code wants a real
      boolean for `Settings.debug`.

    Inputs:
    - `value`: a string from `os.getenv()`, or `None` when the variable is not
      set.

    Outputs:
    - `True` or `False`.

    Example flow:
    - `.env` contains `DEBUG=true`
    - `os.getenv("DEBUG")` returns `"true"`
    - `parse_bool("true")` returns `True`
    """

    if value is None:
        return False

    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_positive_int(value: str | None, default: int) -> int:
    """Parse a positive integer from an environment variable.

    What this function does:
    - Converts strings like `"3"` into the integer `3`.
    - Falls back to `default` for missing, invalid, zero, or negative values.

    Why it exists:
    - `MAX_ITERATIONS` controls how many research rounds the agent may run.
      A value like `0` would make the loop useless, so we only accept positive
      integers.

    Inputs:
    - `value`: the raw environment variable string.
    - `default`: the safe value to use when parsing fails.

    Outputs:
    - A positive integer.
    """

    if value is None or not value.strip():
        return default

    try:
        parsed_value = int(value)
    except ValueError:
        return default

    if parsed_value < 1:
        return default

    return parsed_value


def debug_print(label: str, value=None, enabled: bool = False) -> None:
    """Print verbose debug output only when debug mode is enabled.

    What this function does:
    - Prints a colored `[debug]` label.
    - Optionally prints extra data underneath the label.
    - Pretty-prints dictionaries and lists so complex values are easier to read.

    Why it exists:
    - AI agents have multiple invisible steps: prompt creation, API calls,
      search results, sufficiency decisions, and final synthesis. Debug output
      lets a learner see those steps without changing the final user answer.

    Inputs:
    - `label`: a short description of the step being printed.
    - `value`: optional extra data, such as a dictionary, list, or string.
    - `enabled`: whether debug mode is currently on.

    Outputs:
    - `None`. This function only prints to the terminal.

    Example flow:
    - `debug_print("Generated queries", ["python agents"], enabled=True)`
    - The terminal shows a colored label and a formatted list.
    """

    # Debug output is opt-in. Normal users should see only clean answers, while
    # learners can turn on DEBUG=true to watch the agent think step by step.
    if not enabled:
        return

    print(f"{DEBUG_COLOR_LABEL}[debug] {label}{RESET_COLOR}")

    # Some debug messages only need a label, such as "Calling OpenAI". In that
    # case there is no extra value to print.
    if value is None:
        return

    # Dictionaries and lists can be hard to read on one line, so we pretty-print
    # them as formatted JSON. `default=str` prevents crashes if a value is not
    # directly JSON-serializable.
    if isinstance(value, (dict, list)):
        print(f"{DEBUG_COLOR}{json.dumps(value, indent=2, default=str)}{RESET_COLOR}")
    else:
        print(f"{DEBUG_COLOR}{value}{RESET_COLOR}")


def load_settings() -> Settings:
    """Load environment variables and return a validated `Settings` object.

    What this function does:
    - Reads `.env` into the process.
    - Pulls out the variables this project knows about.
    - Converts strings into useful Python types.
    - Raises `ConfigError` if required API keys are missing.

    Why it exists:
    - Keeping all configuration code in one place makes the rest of the agent
      easier to read. `agent.py` can focus on research behavior instead of
      knowing how `.env` files work.

    Inputs:
    - None directly. It reads from the process environment.

    Outputs:
    - A `Settings` object.

    Example flow:
    - `.env` contains `DEBUG=true` and `MAX_ITERATIONS=3`
    - Docker Compose loads those values into the container
    - `load_settings()` returns `Settings(debug=True, max_iterations=3, ...)`
    """

    # `load_dotenv()` reads a local `.env` file and adds its values to
    # `os.environ`. Docker Compose also loads `.env`, but this call makes the
    # code friendlier if someone runs it in another environment later.
    load_dotenv()

    # `.strip()` removes accidental spaces around values copied into `.env`.
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    debug = parse_bool(os.getenv("DEBUG"))
    max_iterations = parse_positive_int(
        os.getenv("MAX_ITERATIONS"),
        DEFAULT_MAX_ITERATIONS,
    )

    # Collect all missing required settings before raising one error. This is
    # nicer for beginners because they can fix all missing keys at once.
    missing_settings = []
    if not openai_api_key:
        missing_settings.append("OPENAI_API_KEY")
    if not tavily_api_key:
        missing_settings.append("TAVILY_API_KEY")

    if missing_settings:
        missing_names = ", ".join(missing_settings)
        raise ConfigError(
            f"Missing required environment variable(s): {missing_names}. "
            "Copy .env.example to .env and fill in your API keys."
        )

    return Settings(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        openai_model=openai_model,
        debug=debug,
        max_iterations=max_iterations,
    )
