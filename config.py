"""Configuration helpers for the research agent.

This file keeps environment loading and validation in one beginner-friendly
place. The app reads secrets from .env through Docker Compose.
"""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-5.4-mini"


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings used by the agent and tools."""

    openai_api_key: str
    tavily_api_key: str
    openai_model: str = DEFAULT_MODEL


def load_settings() -> Settings:
    """Load and validate settings from environment variables."""

    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

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
    )
