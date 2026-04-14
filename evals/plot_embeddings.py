import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from fashion_rag.search import load_bq_index, load_model


def reduce(embeddings, method="umap"):
    if method == "umap":
        from umap import UMAP
        return UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")
    args = parser.parse_args()

    model, processor = load_model()
    print("Loading embeddings from BQ...")
    image_embs, metadata = load_bq_index()

    print(f"Encoding {len(metadata)} product descriptions...")
    descriptions = [
        f"a picture of {row['productDisplayName']}. A {row['baseColour']} {row['articleType']}."
        for _, row in metadata.iterrows()
    ]
    inputs = processor(text=descriptions, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_embs = model.get_text_features(**inputs).pooler_output
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    text_embs = text_embs.numpy()

    all_embs = np.concatenate([image_embs, text_embs])

    print(f"Running {args.method}...")
    coords = reduce(all_embs, method=args.method)
    img_coords = coords[:len(image_embs)]
    txt_coords = coords[len(image_embs):]

    categories = metadata["articleType"].values
    unique_cats = sorted(set(categories))
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_cats))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    colors = [cmap(cat_to_idx[c]) for c in categories]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    axes[0].scatter(img_coords[:, 0], img_coords[:, 1], c=colors, s=10, alpha=0.7)
    axes[0].set_title("Image embeddings")

    axes[1].scatter(txt_coords[:, 0], txt_coords[:, 1], c=colors, s=10, alpha=0.7)
    axes[1].set_title("Text embeddings (product descriptions)")

    axes[2].scatter(img_coords[:, 0], img_coords[:, 1], c=colors, s=10, alpha=0.5, marker="o")
    axes[2].scatter(txt_coords[:, 0], txt_coords[:, 1], c=colors, s=10, alpha=0.5, marker="x")
    axes[2].set_title("Both (o=image, x=text)")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(cat_to_idx[c]), markersize=6) for c in unique_cats]
    fig.legend(handles, unique_cats, loc="center right", fontsize=7)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    out = f"eval-outputs/embedding_{args.method}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
