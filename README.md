# Research Agent MVP

A beginner-friendly Python CLI research agent that runs inside Docker. It takes
a research question, generates search queries with OpenAI, searches the web with
Tavily, and returns an answer with source links.

## Files

- `main.py`: CLI entry point.
- `agent.py`: research flow orchestration.
- `tools.py`: Tavily web search wrapper.
- `prompts.py`: prompts for query generation and answer writing.
- `config.py`: `.env` loading and settings validation.
- `tests/`: offline pytest tests with mocked API clients.

## Setup

Copy the example environment file and add your real keys:

```bash
cp .env.example .env
```

Required values:

```bash
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
OPENAI_MODEL=gpt-5.4-mini
DEBUG=false
MAX_ITERATIONS=3
```

Build the Docker image:

```bash
docker compose build
```

## Run

Start the interactive CLI. You can ask multiple questions in one session, then
type `exit`, `quit`, or `q` to leave:

```bash
docker compose run --rm research-agent
```

Or pass a question directly:

```bash
docker compose run --rm research-agent python main.py "What are the latest trends in AI safety research?"
```

## Test

Run all tests inside Docker:

```bash
docker compose run --rm research-agent pytest
```

Do not run `pytest`, `python main.py`, or `pip install` directly on the host
machine. Dependencies are intended to live only inside the Docker image.

## Development Notes

The project directory is mounted into the container, so Python code changes do
not require rebuilding. Rebuild only when `requirements.txt`, `Dockerfile`, or
`docker-compose.yml` changes:

```bash
docker compose build
```
