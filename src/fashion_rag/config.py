from upath import UPath

GCP_PROJECT = "fashion-rag"
GCP_REGION = "us-central1"

BQ_DATASET = "fashion"
BQ_METADATA_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.metadata"
BQ_EMBEDDINGS_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.clip_embeddings"

GCS_BUCKET = "fashion-data-500"
BUCKET = f"gs://{GCS_BUCKET}"
IMAGES_DIR = UPath(BUCKET) / "images"

MODEL_NAME = "openai/clip-vit-base-patch32"

COMPONENT_IMAGE = (
    f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT}/containers/fashion-rag-component:latest"
)
