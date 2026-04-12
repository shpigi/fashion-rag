import sys

import numpy as np
import pandas as pd
import torch
from google.cloud import bigquery
from transformers import CLIPModel, CLIPProcessor

GCP_PROJECT = "fashion-rag"
BQ_DATASET = "fashion"
EMBEDDINGS_FILE = "data/embeddings.npz"

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_model():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def load_index():
    data = np.load(EMBEDDINGS_FILE)
    ids = data["ids"]
    embeddings = data["embeddings"]

    client = bigquery.Client(project=GCP_PROJECT)
    metadata = client.query(f"SELECT * FROM `{BQ_DATASET}.metadata`").to_dataframe()
    metadata = metadata.set_index("id").loc[ids].reset_index()
    return embeddings, metadata


def encode_text(query, model, processor):
    inputs = processor(
        text=[f"a photo of {query}"], return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        emb = model.get_text_features(**inputs).pooler_output
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].numpy()


def search(query_emb, embeddings, metadata, k=10):
    scores = embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:k]
    results = metadata.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    return results


def main():
    query = " ".join(sys.argv[1:]) or "red dress"
    model, processor = load_model()
    embeddings, metadata = load_index()
    query_emb = encode_text(query, model, processor)
    results = search(query_emb, embeddings, metadata)

    print(f"\nTop 10 results for: \"{query}\"\n")
    for _, row in results.iterrows():
        print(f"  {row['score']:.3f}  {row['id']}  {row['baseColour']} {row['articleType']} ({row['gender']}, {row['season']})")


if __name__ == "__main__":
    main()
