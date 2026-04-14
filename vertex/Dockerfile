FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --extra ml --no-install-project

COPY src/ src/
COPY evals/ evals/

RUN uv sync --frozen --no-dev --extra ml

ENV PATH="/app/.venv/bin:$PATH"
