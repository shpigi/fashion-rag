import sys

import torch
from google.cloud import bigquery
from transformers import CLIPModel, CLIPProcessor

GCP_PROJECT = "fashion-rag"
BQ_DATASET = "fashion"
BQ_EMBEDDINGS_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.clip_embeddings"
BQ_METADATA_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.metadata"

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_model():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def encode_text(query, model, processor):
    inputs = processor(
        text=[f"a photo of {query}"], return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        emb = model.get_text_features(**inputs).pooler_output
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].numpy()


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
