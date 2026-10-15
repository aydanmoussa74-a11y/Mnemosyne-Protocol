# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps (minimal; sqlite is built into the python image).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps. Use the full "all" extra so OpenAI/Anthropic/Ollama/vLLM adapters
# and numpy/tiktoken are available. Install mnemosyne last so it can use them.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ".[all]"

# Application code.
COPY mnemosyne/ ./mnemosyne/
COPY tests/ ./tests/

# Default data directory (persisted via docker-compose volume).
ENV MNEMOSYNE_STORAGE_PATH=/data/mnemosyne
ENV MNEMOSYNE_DB_ECHO=false

# Healthcheck against the FastAPI service.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

EXPOSE 8000

# Run the FastAPI microservice with uvicorn.
CMD ["uvicorn", "mnemosyne.server:app", "--host", "0.0.0.0", "--port", "8000"]
