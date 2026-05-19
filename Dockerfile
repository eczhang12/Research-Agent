# syntax=docker/dockerfile:1

# This file teaches Docker how to build the container image for the app.
# A container image is like a small, repeatable Linux environment that includes
# Python and the app dependencies, so your host machine stays clean.

# Python runtime for the research-agent service. The `slim` image is smaller
# than the full Python image but still beginner-friendly for this project.
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

# This is the default command when the container starts. Docker Compose can
# override it, for example when running `pytest`.
CMD ["python", "main.py"]
