import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage

from fashion_rag.search import encode_text, load_bq_index, load_model, local_search

K = 5


def build_confusion(metadata, embeddings, model, processor, field):
    values = sorted(metadata[field].unique())
    val_to_idx = {v: i for i, v in enumerate(values)}
    n = len(values)

    confusion = np.zeros((n, n))
    query_counts = np.zeros(n)

    other_field = "baseColour" if field == "articleType" else "articleType"
    other_values = sorted(metadata[other_field].unique())

    for val in values:
        for other in other_values:
            available = len(metadata[(metadata[field] == val) & (metadata[other_field] == other)])
            if available == 0:
                continue

            if field == "articleType":
                query = f"A product image showing a {other} {val}"
            else:
                query = f"A product image showing a {val} {other}"

            query_emb = encode_text(query, model, processor)
            results = local_search(query_emb, embeddings, metadata, k=K)

            for _, row in results.iterrows():
                confusion[val_to_idx[val], val_to_idx[row[field]]] += 1
            query_counts[val_to_idx[val]] += len(results)

    row_sums = query_counts[:, None]
    row_sums[row_sums == 0] = 1
    confusion_pct = confusion / row_sums

    return confusion_pct, values


def plot_confusion(confusion_pct, labels, field, out):
    # Cluster by row similarity, apply same order to both axes
    Z = linkage(confusion_pct, method="ward")
    order = leaves_list(Z)
    confusion_pct = confusion_pct[order][:, order]
    labels = [labels[i] for i in order]
    n = len(labels)

    masked = np.ma.masked_where(confusion_pct == 0, confusion_pct)
    cmap = plt.colormaps.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="0.92")

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"{field} of retrieved images")
    ax.set_ylabel(f"{field} in text query")
    ax.set_title(f"Retrieval confusion by {field} (fraction of top-{K} results)")

    ax.set_xticks([x - 0.5 for x in range(n + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(n + 1)], minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.3, alpha=0.5)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            val = confusion_pct[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if val > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", choices=["articleType", "baseColour", "both"], default="both")
    args = parser.parse_args()

    model, processor = load_model()
    print("Loading embeddings from BQ...")
    embeddings, metadata = load_bq_index()

    fields = ["articleType", "baseColour"] if args.field == "both" else [args.field]

    for field in fields:
        print(f"Building {field} confusion...")
        confusion_pct, labels = build_confusion(metadata, embeddings, model, processor, field)
        out = f"eval-outputs/confusion_{field}.png"
        plot_confusion(confusion_pct, labels, field, out)


if __name__ == "__main__":
    main()
