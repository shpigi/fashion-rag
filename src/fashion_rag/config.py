from upath import UPath

GCP_PROJECT = "fashion-rag"
GCP_REGION = "us-central1"

BQ_DATASET = "fashion"
BQ_METADATA_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.metadata"
BQ_EMBEDDINGS_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.clip_embeddings"

BUCKET = "gs://fashion-data-500"
IMAGES_DIR = UPath(BUCKET) / "images"

MODEL_NAME = "openai/clip-vit-base-patch32"
