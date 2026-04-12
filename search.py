import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
DATA_DIR = Path("data")
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"
METADATA_CSV = DATA_DIR / "metadata.csv"


def load_index():
    data = np.load(EMBEDDINGS_FILE)
    metadata = pd.read_csv(METADATA_CSV).set_index("id").loc[data["ids"]].reset_index()
    return data["embeddings"], metadata


def search(query_emb, embeddings, metadata, k=10):
    scores = embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:k]
    results = metadata.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    return results


def main():
    query = " ".join(sys.argv[1:]) or "red dress"

    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = processor(
        text=[f"a photo of {query}"], return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        emb = model.get_text_features(**inputs).pooler_output
    emb = emb / emb.norm(dim=-1, keepdim=True)
    query_emb = emb[0].numpy()

    embeddings, metadata = load_index()
    results = search(query_emb, embeddings, metadata)

    print(f"\nTop 10 results for: \"{query}\"\n")
    for _, row in results.iterrows():
        print(f"  {row['score']:.3f}  {row['id']}  {row['baseColour']} {row['articleType']} ({row['gender']}, {row['season']})")


if __name__ == "__main__":
    main()
