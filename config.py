"""Configuration helpers for the research agent.

This file keeps environment loading and validation in one beginner-friendly
place. The app reads secrets from .env through Docker Compose.
"""

from dataclasses import dataclass
import json
import os

from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-5.4-mini"
DEBUG_COLOR_LABEL = "\x1B[33m"
DEBUG_COLOR = "\033[36m"
RESET_COLOR = "\033[0m"


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings used by the agent and tools."""

    openai_api_key: str
    tavily_api_key: str
    openai_model: str = DEFAULT_MODEL
    debug: bool = False


def parse_bool(value: str | None) -> bool:
    """Convert common environment-style truthy values into a boolean."""

    if value is None:
        return False

    return value.strip().lower() in {"1", "true", "yes", "on"}


def debug_print(label: str, value=None, enabled: bool = False) -> None:
    """Print verbose debug output only when debug mode is enabled.

    Parameters:
    - label: a short description of the step being printed.
    - value: optional extra data, such as a dictionary, list, or string.

    Return value:
    - None. This function only prints to the terminal.

    How this fits the architecture:
    The research agent has several request/response steps. This helper gives
    all modules one consistent way to show those steps while keeping normal
    output clean when debug mode is off.
    """

    if not enabled:
        return

    print(f"{DEBUG_COLOR_LABEL}[debug] {label}{RESET_COLOR}")

    if value is None:
        return

    if isinstance(value, (dict, list)):
        print(f"{DEBUG_COLOR}{json.dumps(value, indent=2, default=str)}{RESET_COLOR}")
    else:
        print(f"{DEBUG_COLOR}{value}{RESET_COLOR}")


def load_settings() -> Settings:
    """Load and validate settings from environment variables."""

    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    debug = parse_bool(os.getenv("DEBUG"))

    missing = []
    if not openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not tavily_api_key:
        missing.append("TAVILY_API_KEY")

    if missing:
        names = ", ".join(missing)
        raise ConfigError(
            f"Missing required environment variable(s): {names}. "
            "Copy .env.example to .env and fill in your API keys."
        )

    return Settings(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        openai_model=openai_model,
        debug=debug,
    )
