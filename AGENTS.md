# AGENTS.md

## Project Goal
Build a beginner-friendly Python research agent that can answer open-ended questions using web search and source-grounded synthesis.

The project should run and be tested inside Docker.

## Architecture
Keep the project modular:

- `main.py`: CLI entry point
- `agent.py`: research loop and orchestration
- `tools.py`: external tools like web search
- `prompts.py`: system prompts and prompt templates
- `config.py`: environment variables and settings
- `tests/`: lightweight pytest tests

## Docker Rules
- The app must run inside Docker.
- Do not assume dependencies exist on the host machine.
- Use `requirements.txt` for Python dependencies.
- Use `.env` for secrets and environment variables.
- Include `.env.example` with placeholder values.
- Use `docker-compose.yml` for common commands.
- Mount the local project directory into the container for development.
- Code changes should not require rebuilding the image.
- Rebuild only when dependencies or Docker configuration changes.

## Expected Commands
The app should run with:

```bash
docker compose run --rm research-agent
```


## Testing and Execution Rules

Codex must test and run this project inside Docker only.

Do not run project commands directly on the native host machine.

Allowed commands:

```bash
docker compose run --rm research-agent pytest
docker compose run --rm research-agent python main.py
docker compose run --rm research-agent
docker compose build
```