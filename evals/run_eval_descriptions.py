DESCRIPTION = """\
Description-based retrieval eval: for each of the 500 items, encode the
productDisplayName as a CLIP text query and check whether the item's own
image appears in the top-K results.

This measures how well CLIP aligns a product's text description with its
image in the shared embedding space. Unlike the category eval, this targets
exact item retrieval rather than category-level matching.

Metrics:
  MRR@K  — Mean Reciprocal Rank: average of 1/rank of the item's own image.
           MRR=1 means every item is the top result for its own description.
  Hit@1  — Fraction of items where the top result is the item itself.
  Hit@K  — Fraction of items found anywhere in the top-K results.

Breakdown by articleType shows which categories have the most visually
distinctive items (high MRR) vs categories where items look too similar
for CLIP to distinguish by description alone (low MRR).
"""

import argparse

import pandas as pd

from fashion_rag.search import encode_texts, load_bq_index, load_model, local_search


def reciprocal_rank(results, target_id):
    for rank, (_, row) in enumerate(results.iterrows(), 1):
        if row["id"] == target_id:
            return 1.0 / rank
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    model, processor = load_model()
    print("Loading embeddings and metadata from BQ...")
    embeddings, metadata = load_bq_index()

    k = args.k

    queries = [
        f"{row['productDisplayName']}"
        for _, row in metadata.iterrows()
    ]
    all_embs = encode_texts(queries, model, processor)

    results_list = []
    for (_, item), query, query_emb in zip(metadata.iterrows(), queries, all_embs):
        topk = local_search(query_emb, embeddings, metadata, k=k)
        rr = reciprocal_rank(topk, item["id"])
        results_list.append({
            "id": item["id"],
            "query": query[:60],
            "articleType": item["articleType"],
            "baseColour": item["baseColour"],
            "rr": rr,
            "rank": int(1 / rr) if rr > 0 else None,
        })

    df = pd.DataFrame(results_list)

    lines = []
    lines.append(f"Description-based retrieval eval (K={k}, {len(df)} items)\n")

    by_type = df.groupby("articleType")["rr"].mean().sort_values(ascending=False)
    lines.append(f"{'articleType':<25} {'MRR':>6} {'Count':>5}")
    lines.append("-" * 40)
    for atype, mrr in by_type.items():
        count = len(df[df["articleType"] == atype])
        lines.append(f"{atype:<25} {mrr:>6.3f} {count:>5}")

    lines.append(f"\n{'=' * 40}")
    lines.append(f"MRR@{k}:              {df['rr'].mean():.3f}")
    lines.append(f"Hit@1:              {(df['rr'] == 1.0).mean():.3f}")
    lines.append(f"Hit@{k}:              {(df['rr'] > 0).mean():.3f}")

    report = "\n".join(lines)
    print(report)

    out = "eval-outputs/eval_descriptions.txt"
    with open(out, "w") as f:
        f.write(report + "\n\n" + DESCRIPTION)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
