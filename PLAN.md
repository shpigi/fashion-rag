# Fashion RAG — Development Plan

Local-first development. Get everything working on local data before touching GCP.

## Phase 1: Local CLIP Embedding + Search Validation

**Goal:** Prove that CLIP retrieval works on our 500-image subset using search terms derived from metadata.

- [x] `embed.py` — Load CLIP ViT-B/32 (HuggingFace `transformers`), batch-encode all 500 images, save embeddings to a local numpy/parquet file alongside product IDs
- [x] `search.py` — Load saved embeddings, encode a text query with CLIP text encoder, compute cosine similarity, return top-K results

**Output:** Embeddings file, search function, eval report with per-query and aggregate Precision@K.

## Phase 2: Local Streamlit App

- [ ] `app.py` — Text input, encode with CLIP, search local embeddings, display top-K images from `data/images/` with metadata match indicators and Precision@K
- [ ] Predefined eval queries dropdown (from Phase 1 eval set)
- [ ] Free-text query support with keyword-based metadata matching for eval

## Phase 3: GCP Migration

- [ ] Upload images to GCS bucket
- [ ] Load metadata + embeddings into BigQuery with vector index
- [ ] `pipeline.py` — Vertex AI Pipeline wrapping the CLIP embedding step as a KFP component
- [ ] Switch Streamlit app to query BigQuery VECTOR_SEARCH instead of local numpy

## Phase 4: Polish for Demo

- [ ] Clean Streamlit UI (image cards, distance scores, green/red match indicators)
- [ ] Predefined eval queries with hand-labeled expected attributes
- [ ] Prepare talking points for 4 design decisions (CLIP vs Google embeddings, pipeline as Vertex step, BQ as vector store, metadata-based eval)
