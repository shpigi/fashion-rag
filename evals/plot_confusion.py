import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage

from fashion_rag.search import encode_text, load_bq_index, load_model, local_search

K = 5


def main():
    model, processor = load_model()
    print("Loading embeddings from BQ...")
    embeddings, metadata = load_bq_index()

    article_types = sorted(metadata["articleType"].unique())
    type_to_idx = {t: i for i, t in enumerate(article_types)}
    n = len(article_types)

    # For each type, query with each available colour and aggregate retrieved types
    confusion = np.zeros((n, n))
    query_counts = np.zeros(n)

    colours = sorted(metadata["baseColour"].unique())
    for atype in article_types:
        for colour in colours:
            available = len(metadata[(metadata["articleType"] == atype) & (metadata["baseColour"] == colour)])
            if available == 0:
                continue

            query = f"A product image showing a {colour} {atype}"
            query_emb = encode_text(query, model, processor)
            results = local_search(query_emb, embeddings, metadata, k=K)

            for _, row in results.iterrows():
                confusion[type_to_idx[atype], type_to_idx[row["articleType"]]] += 1
            query_counts[type_to_idx[atype]] += len(results)

    # Normalize rows to fractions
    row_sums = query_counts[:, None]
    row_sums[row_sums == 0] = 1
    confusion_pct = confusion / row_sums

    # Cluster by row similarity, apply same order to both axes
    Z = linkage(confusion_pct, method="ward")
    order = leaves_list(Z)
    confusion_pct = confusion_pct[order][:, order]
    article_types = [article_types[i] for i in order]

    # ==============================================================================
    # Plot

    # Show zero cells as grey
    masked = np.ma.masked_where(confusion_pct == 0, confusion_pct)
    cmap = plt.colormaps.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="0.92")

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(article_types, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(article_types, fontsize=8)
    ax.set_xlabel("articleType of retrieved images")
    ax.set_ylabel("articleType in text query")
    ax.set_title(f"Retrieval confusion (fraction of top-{K} results)")

    ax.set_xticks([x - 0.5 for x in range(n + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(n + 1)], minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.3, alpha=0.5)
    ax.tick_params(which="minor", length=0)

    # Annotate cells with values > 0.05
    for i in range(n):
        for j in range(n):
            val = confusion_pct[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if val > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    out = "eval-outputs/retrieval_confusion.png"
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
