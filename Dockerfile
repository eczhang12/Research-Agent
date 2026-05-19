# syntax=docker/dockerfile:1

# Python runtime for the research-agent service.
FROM python:3.12-slim

# Keep Python logs readable in Docker and avoid writing .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies are installed before the source code is copied so Docker can
# reuse this layer until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app for image completeness. docker-compose.yml also bind-mounts the
# project directory during development so code edits do not require rebuilds.
COPY . .

CMD ["python", "main.py"]
