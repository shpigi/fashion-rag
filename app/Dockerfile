FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --frozen --no-dev --extra app

COPY app/ app/

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080

CMD ["streamlit", "run", "app/app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
