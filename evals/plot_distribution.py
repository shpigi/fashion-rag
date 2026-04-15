from typing import BinaryIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery

from fashion_rag.config import BQ_METADATA_TABLE, GCP_PROJECT


def plot_distribution(metadata: pd.DataFrame, out_path: str | BinaryIO) -> None:
    ct = pd.crosstab(metadata["articleType"], metadata["baseColour"], normalize="all")
    types = ct.index.tolist()
    colours = ct.columns.tolist()
    matrix = ct.values

    masked = np.ma.masked_where(matrix == 0, matrix)
    cmap = plt.colormaps.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="0.92")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(masked, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(colours)))
    ax.set_yticks(range(len(types)))
    ax.set_xticklabels(colours, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(types, fontsize=8)
    ax.set_xlabel("baseColour")
    ax.set_ylabel("articleType")
    ax.set_title("Dataset distribution: articleType x baseColour (fraction of total)")

    ax.set_xticks([x - 0.5 for x in range(len(colours) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(types) + 1)], minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.3, alpha=0.5)
    ax.tick_params(which="minor", length=0)

    for i in range(len(types)):
        for j in range(len(colours)):
            val = matrix[i, j]
            if val > 0.001:
                ax.text(
                    j,
                    i,
                    f"{val:.1%}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if val > 0.02 else "black",
                )

    fig.colorbar(ax.images[0], ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, format="png")
    plt.close()


def main() -> None:
    client = bigquery.Client(project=GCP_PROJECT)
    metadata = client.query(f"SELECT * FROM `{BQ_METADATA_TABLE}`").to_dataframe()

    out = "eval-outputs/dataset_distribution.png"
    plot_distribution(metadata, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
