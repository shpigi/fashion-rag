DESCRIPTION = """\
Category retrieval eval: for each (baseColour, articleType) combination in the
metadata, encode "{colour} {articleType}" as a CLIP text query and retrieve
the top-K most similar images from the embedding index.

Metrics:
  MRR@K         — Mean Reciprocal Rank: average of 1/rank of the first result
                  matching both colour and type. Higher = correct items appear
                  earlier in the results.
  MRR@K(colour) — MRR considering only colour match.
  MRR@K(type)   — MRR considering only articleType match.
  Recall@K      — Fraction of available matching items found in top-K,
                  averaged across queries. Denominator is min(K, available)
                  so rare categories are not penalised.
"""

import argparse

import numpy as np
import pandas as pd

from fashion_rag.search import encode_texts, load_bq_index, load_model, local_search

MIN_ITEMS_FOR_QUERY = 1


def reciprocal_rank(matches):
    hits = matches.values.nonzero()[0]
    return 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0


def run_category_eval(embeddings, metadata, model, processor, k=10):
    combos = metadata.groupby(["baseColour", "articleType"]).size().reset_index(name="count")
    combos = combos[combos["count"] >= MIN_ITEMS_FOR_QUERY]

    queries = []
    for _, row in combos.iterrows():
        colour, atype, count = row["baseColour"], row["articleType"], row["count"]
        queries.append(
            {
                "query": f"{colour} {atype}",
                "expected": {"baseColour": colour, "articleType": atype},
                "available": count,
            }
        )

    query_texts = [q["query"] for q in queries]
    all_embs = encode_texts(query_texts, model, processor)

    per_query = []
    for q, query_emb in zip(queries, all_embs):
        topk = local_search(query_emb, embeddings, metadata, k=k)
        colour_match = topk["baseColour"].str.lower() == q["expected"]["baseColour"].lower()
        type_match = topk["articleType"].str.lower() == q["expected"]["articleType"].lower()
        both_match = colour_match & type_match

        per_query.append(
            {
                "query": q["query"],
                "available": q["available"],
                "rr": reciprocal_rank(both_match),
                "rr_colour": reciprocal_rank(colour_match),
                "rr_type": reciprocal_rank(type_match),
                "r@k": both_match.sum() / min(k, q["available"]),
            }
        )

    df = pd.DataFrame(per_query)
    summary = {
        "mrr": float(df["rr"].mean()),
        "mrr_colour": float(df["rr_colour"].mean()),
        "mrr_type": float(df["rr_type"].mean()),
        "recall": float(df["r@k"].mean()),
        "num_queries": len(df),
    }

    type_confusion = _build_confusion(queries, all_embs, metadata, embeddings, "articleType")
    colour_confusion = _build_confusion(queries, all_embs, metadata, embeddings, "baseColour")

    return df, summary, type_confusion, colour_confusion


def _build_confusion(queries, all_embs, metadata, embeddings, field):
    values = sorted(metadata[field].unique())
    val_to_idx = {v: i for i, v in enumerate(values)}
    n = len(values)
    confusion = np.zeros((n, n))
    query_counts = np.zeros(n)

    for q, query_emb in zip(queries, all_embs):
        val = q["expected"][field]
        results = local_search(query_emb, embeddings, metadata, k=5)
        for _, row in results.iterrows():
            confusion[val_to_idx[val], val_to_idx[row[field]]] += 1
        query_counts[val_to_idx[val]] += len(results)

    row_sums = query_counts[:, None]
    row_sums[row_sums == 0] = 1
    return {"labels": values, "matrix": confusion / row_sums}


def plot_confusion(confusion, title, out_path):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage

    labels = confusion["labels"]
    matrix = confusion["matrix"]
    n = len(labels)

    Z = linkage(matrix, method="ward")
    order = leaves_list(Z)
    matrix = matrix[order][:, order]
    labels = [labels[i] for i in order]

    masked = np.ma.masked_where(matrix == 0, matrix)
    cmap = plt.colormaps.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="0.92")

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("retrieved")
    ax.set_ylabel("queried")
    ax.set_title(title)

    ax.set_xticks([x - 0.5 for x in range(n + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(n + 1)], minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.3, alpha=0.5)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0.05:
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if val > 0.5 else "black",
                )

    fig.colorbar(ax.images[0], ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, format="png")
    plt.close()


def format_report(df, summary, k):
    df = df.sort_values("rr", ascending=False)
    lines = []
    lines.append(
        f"{'Query':<30} {'Avail':>5} {f'RR@{k}':>6} {'RR_col':>6} {'RR_typ':>6} {f'R@{k}':>6}"
    )
    lines.append("-" * 64)
    for _, r in df.iterrows():
        lines.append(
            f"{r['query']:<30} {r['available']:>5} {r['rr']:>6.2f} "
            f"{r['rr_colour']:>6.2f} {r['rr_type']:>6.2f} {r['r@k']:>6.2f}"
        )
    lines.append(f"\n{'=' * 64}")
    lines.append(f"MRR@{k}:                   {summary['mrr']:.3f}")
    lines.append(f"MRR@{k} (colour):          {summary['mrr_colour']:.3f}")
    lines.append(f"MRR@{k} (type):            {summary['mrr_type']:.3f}")
    lines.append(f"Mean Recall@{k}:           {summary['recall']:.3f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    model, processor = load_model()
    print("Loading embeddings and metadata from BQ...")
    embeddings, metadata = load_bq_index()

    df, summary, type_conf, colour_conf = run_category_eval(
        embeddings, metadata, model, processor, k=args.k
    )

    report = format_report(df, summary, args.k)
    print(report)

    out = "eval-outputs/eval_categories.txt"
    with open(out, "w") as f:
        f.write(report + "\n\n" + DESCRIPTION)
    print(f"\nSaved to {out}")

    plot_confusion(
        type_conf,
        "Retrieval confusion by articleType (top-5)",
        "eval-outputs/confusion_articleType.png",
    )
    plot_confusion(
        colour_conf,
        "Retrieval confusion by baseColour (top-5)",
        "eval-outputs/confusion_baseColour.png",
    )


if __name__ == "__main__":
    main()
