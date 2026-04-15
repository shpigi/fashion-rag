import numpy as np
import pandas as pd

from fashion_rag.search import local_search


def _make_index(n=50, dim=8):
    """Create a small synthetic embedding index."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    metadata = pd.DataFrame({"id": range(n), "label": [f"item_{i}" for i in range(n)]})
    return embeddings, metadata


def test_local_search_returns_k_results():
    embeddings, metadata = _make_index()
    query = embeddings[0]
    results = local_search(query, embeddings, metadata, k=5)
    assert len(results) == 5


def test_local_search_top_hit_is_query_itself():
    embeddings, metadata = _make_index()
    query = embeddings[3]
    results = local_search(query, embeddings, metadata, k=5)
    assert results.iloc[0]["id"] == 3


def test_local_search_scores_are_descending():
    embeddings, metadata = _make_index()
    query = embeddings[0]
    results = local_search(query, embeddings, metadata, k=10)
    scores = results["score"].values
    assert (scores[:-1] >= scores[1:]).all()


def test_local_search_top_score_near_one_for_exact_match():
    embeddings, metadata = _make_index()
    query = embeddings[7]
    results = local_search(query, embeddings, metadata, k=1)
    assert results.iloc[0]["score"] > 0.99


def test_local_search_k_larger_than_index():
    embeddings, metadata = _make_index(n=5)
    query = embeddings[0]
    results = local_search(query, embeddings, metadata, k=20)
    assert len(results) == 5
