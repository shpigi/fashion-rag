import sys

import numpy as np
import torch
from google.cloud import bigquery
from transformers import CLIPModel, CLIPProcessor

from fashion_rag.config import (
    BQ_EMBEDDINGS_TABLE,
    BQ_METADATA_TABLE,
    GCP_PROJECT,
    MODEL_NAME,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def encode_texts(queries, model, processor, batch_size=32):
    all_embs = []
    for i in range(0, len(queries), batch_size):
        batch = [f"a photo of {q}" for q in queries[i : i + batch_size]]
        inputs = processor(
            text=batch, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            emb = model.get_text_features(**inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs)


def encode_text(query, model, processor):
    return encode_texts([query], model, processor)[0]


def load_bq_index():
    client = bigquery.Client(project=GCP_PROJECT)
    metadata = client.query(f"""
        SELECT e.id, e.embedding, m.* EXCEPT(id)
        FROM `{BQ_EMBEDDINGS_TABLE}` e
        JOIN `{BQ_METADATA_TABLE}` m ON e.id = m.id
    """).to_dataframe()
    embeddings = np.stack(metadata.pop("embedding").values)
    return embeddings, metadata


def local_search(query_emb, embeddings, metadata, k=10):
    scores = embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:k]
    results = metadata.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    return results


def search(query_emb, k=10):
    client = bigquery.Client(project=GCP_PROJECT)
    emb_str = ", ".join(str(float(x)) for x in query_emb)
    sql = f"""
    SELECT
        base.id,
        distance,
        meta.* EXCEPT(id)
    FROM VECTOR_SEARCH(
        TABLE `{BQ_EMBEDDINGS_TABLE}`,
        'embedding',
        (SELECT [{emb_str}] AS embedding),
        top_k => {k},
        distance_type => 'COSINE'
    )
    JOIN `{BQ_METADATA_TABLE}` meta ON base.id = meta.id
    ORDER BY distance ASC
    """
    df = client.query(sql).to_dataframe()
    df["score"] = 1.0 - df["distance"]
    df = df.drop(columns=["distance"])
    return df


def main():
    query = " ".join(sys.argv[1:]) or "red dress"
    model, processor = load_model()
    query_emb = encode_text(query, model, processor)
    results = search(query_emb)

    print(f"\nTop 10 results for: \"{query}\"\n")
    for _, row in results.iterrows():
        print(f"  {row['score']:.3f}  {row['id']}  {row['baseColour']} {row['articleType']} ({row['gender']}, {row['season']})")


if __name__ == "__main__":
    main()
