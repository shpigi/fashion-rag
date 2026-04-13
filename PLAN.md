# Fashion RAG — Development Plan

Local-first development. Get everything working on local data before touching GCP.

## Phase 1: Local CLIP Embedding + Search Validation

**Goal:** Prove that CLIP retrieval works on our 500-image subset using search terms derived from metadata.

- [x] `embed.py` — Load CLIP ViT-B/32 (HuggingFace `transformers`), batch-encode all 500 images, save embeddings to a local numpy/parquet file alongside product IDs
- [x] `search.py` — Load saved embeddings, encode a text query with CLIP text encoder, compute cosine similarity, return top-K results

**Output:** Embeddings file, search function, eval report with per-query and aggregate Precision@K.

## Phase 2: Local Streamlit App

- [x] `app.py` — Text input, encode with CLIP, search local embeddings, display top-K images from `data/images/` with metadata match indicators and Precision@K
- [x] Predefined eval queries dropdown (from Phase 1 eval set)
- [x] Free-text query support with keyword-based metadata matching for eval

## Phase 3: GCP Migration

- [x] Upload images to GCS bucket
- [x] Load metadata fro GCP
- [x] App loads images from gcs
- [x] Upload embeddings into BigQuery with vector index
- [x] Switch Streamlit app to query BigQuery VECTOR_SEARCH instead of local numpy


## Phase 4: Pipeline

- [x] `pipeline.py` — Vertex AI Pipeline wrapping the CLIP embedding step as a KFP component

## Phase 5: Eval

- [x] How well does clip based image search match item type and colour

