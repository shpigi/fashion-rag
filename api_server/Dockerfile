FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --frozen --no-dev --extra api

COPY api_server/ api_server/

# Pre-download model weights into the image to avoid cold-start downloads
RUN .venv/bin/python -c "from transformers import CLIPModel, CLIPProcessor; from fashion_rag.config import MODEL_NAME; CLIPModel.from_pretrained(MODEL_NAME); CLIPProcessor.from_pretrained(MODEL_NAME)"

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080

CMD ["python", "api_server/server.py"]
