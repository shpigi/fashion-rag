import sys

import numpy as np
import pandas as pd
import torch
from google.cloud import bigquery
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from fashion_rag.config import (
    BQ_EMBEDDINGS_TABLE,
    BQ_METADATA_TABLE,
    GCP_PROJECT,
    MODEL_NAME,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def encode_texts(
    queries: list[str], model: CLIPModel, processor: CLIPProcessor, batch_size: int = 32
) -> np.ndarray:
    all_embs = []
    for i in range(0, len(queries), batch_size):
        batch = [f"a photo of {q}" for q in queries[i : i + batch_size]]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(
            DEVICE
        )
        with torch.no_grad():
            emb = model.get_text_features(**inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs)


def encode_text(query: str, model: CLIPModel, processor: CLIPProcessor) -> np.ndarray:
    return encode_texts([query], model, processor)[0]


def encode_image(image: Image.Image, model: CLIPModel, processor: CLIPProcessor) -> np.ndarray:
    inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs).pooler_output
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


def load_bq_index() -> tuple[np.ndarray, pd.DataFrame]:
    client = bigquery.Client(project=GCP_PROJECT)
    metadata = client.query(f"""
        SELECT e.id, e.embedding, m.* EXCEPT(id)
        FROM `{BQ_EMBEDDINGS_TABLE}` e
        JOIN `{BQ_METADATA_TABLE}` m ON e.id = m.id
    """).to_dataframe()
    embeddings = np.stack(metadata.pop("embedding").values)
    return embeddings, metadata


def local_search(
    query_emb: np.ndarray, embeddings: np.ndarray, metadata: pd.DataFrame, k: int = 10
) -> pd.DataFrame:
    scores = embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:k]
    results = metadata.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    return results


def get_metadata_values() -> dict[str, list[str]]:
    """Return sorted unique values for key metadata columns."""
    client = bigquery.Client(project=GCP_PROJECT)
    values = {}
    for col in ("baseColour", "articleType"):
        rows = client.query(
            f"SELECT DISTINCT `{col}` FROM `{BQ_METADATA_TABLE}`"
            f" WHERE `{col}` IS NOT NULL ORDER BY 1"
        ).result()
        values[col] = [row[0] for row in rows]
    return values


def search_by_id(item_id: int, k: int = 10) -> pd.DataFrame:
    client = bigquery.Client(project=GCP_PROJECT)
    sql = f"""
    SELECT
        base.id,
        distance,
        meta.* EXCEPT(id)
    FROM VECTOR_SEARCH(
        TABLE `{BQ_EMBEDDINGS_TABLE}`,
        'embedding',
        (SELECT embedding FROM `{BQ_EMBEDDINGS_TABLE}` WHERE id = {item_id}),
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


def search(query_emb: np.ndarray, k: int = 10) -> pd.DataFrame:
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


def main() -> None:
    query = " ".join(sys.argv[1:]) or "red dress"
    model, processor = load_model()
    query_emb = encode_text(query, model, processor)
    results = search(query_emb)

    print(f'\nTop 10 results for: "{query}"\n')
    for _, row in results.iterrows():
        r = row
        print(
            f"  {r['score']:.3f}  {r['id']}  {r['baseColour']} {r['articleType']}"
            f" ({r['gender']}, {r['season']})"
        )


if __name__ == "__main__":
    main()
