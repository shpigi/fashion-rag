# CLAUDE.md

## Project overview

Cross-modal fashion product retrieval system. Users search a 500-image product catalog
by text description or image upload; the system returns visually matching products using
CLIP embeddings and BigQuery vector search. Deployed on GCP (Cloud Run, Vertex AI).

## Architecture

```
src/fashion_rag/     Core library (config, embedding, search)
api_server/          FastAPI REST service (text/image search endpoints)
app/                 Streamlit web UI
skill/               Claude Code skill (CLI client + SKILL.md for /fashion-search)
vertex/              Vertex AI Pipeline (KFP components for batch embedding)
evals/               Evaluation scripts (retrieval metrics, visualizations)
data/                Local data (500 images, metadata.csv, embeddings.npz)
tests/               pytest unit tests
```

**Data flow:** Images -> CLIP ViT-B/32 -> 512-dim embeddings -> BigQuery -> vector search

**GCP services:** BigQuery (vector store + metadata), GCS (image storage),
Vertex AI / KFP (embedding pipeline), Cloud Run (API + app), Artifact Registry (containers)

## Package structure

Python package `fashion_rag` built with hatchling. Source in `src/fashion_rag/`.
Dependencies managed with `uv`. Optional extras: `ml`, `app`, `api`, `vertex`, `eval`,
`test`, `dev`.

## Key commands

```
make api              # Start FastAPI dev server (port 8080)
make app              # Start Streamlit UI (needs API running)
make serve            # Start both API + app
make embed            # Run local CLIP embedding
make search           # Test search CLI
make pipeline         # Build component image + submit Vertex AI pipeline
make deploy-api       # Build + deploy API to Cloud Run
make deploy-app       # Build + deploy app to Cloud Run
```

## Code style

- Ruff for linting and formatting (config in pyproject.toml: line-length 100, py312)
- Pre-commit hook enforces `ruff check` and `ruff format`
- Minimal comments. Only where genuinely needed
- Keep code direct and simple; no unnecessary abstractions
