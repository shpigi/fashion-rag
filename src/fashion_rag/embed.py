import numpy as np
import torch
from google.cloud import bigquery
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from upath import UPath

from fashion_rag.config import (
    BQ_EMBEDDINGS_TABLE,
    BQ_METADATA_TABLE,
    GCP_PROJECT,
    IMAGES_DIR,
    MODEL_NAME,
)

BATCH_SIZE = 32
NUM_WORKERS = 4

BQ_SCHEMA = [
    bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
]


class ImageDataset(Dataset):
    def __init__(self, image_paths: list[UPath]) -> None:
        self.paths = image_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Image.Image:
        with self.paths[idx].open("rb") as f:
            return Image.open(f).convert("RGB")


def get_ids_to_embed(recreate: bool = True) -> list[int]:
    client = bigquery.Client(project=GCP_PROJECT)
    all_ids = [
        row.id for row in client.query(f"SELECT id FROM `{BQ_METADATA_TABLE}` ORDER BY id").result()
    ]

    if recreate:
        return all_ids

    try:
        embedded = {
            row.id for row in client.query(f"SELECT id FROM `{BQ_EMBEDDINGS_TABLE}`").result()
        }
    except Exception:
        embedded = set()

    missing = sorted(set(all_ids) - embedded)
    print(f"{len(all_ids)} in metadata, {len(embedded)} already embedded, {len(missing)} to embed")
    return missing


def write_embeddings(
    ids: list[int], embeddings: np.ndarray, disposition: str = "WRITE_APPEND"
) -> None:
    client = bigquery.Client(project=GCP_PROJECT)
    rows = [{"id": int(pid), "embedding": emb.tolist()} for pid, emb in zip(ids, embeddings)]
    job_config = bigquery.LoadJobConfig(
        write_disposition=disposition,
        schema=BQ_SCHEMA,
    )
    job = client.load_table_from_json(rows, BQ_EMBEDDINGS_TABLE, job_config=job_config)
    job.result()
    print(f"Wrote {len(rows)} rows to {BQ_EMBEDDINGS_TABLE} ({disposition})")


def delete_embeddings_table() -> None:
    client = bigquery.Client(project=GCP_PROJECT)
    client.delete_table(BQ_EMBEDDINGS_TABLE, not_found_ok=True)
    print(f"Deleted {BQ_EMBEDDINGS_TABLE}")


def embed_images(ids: list[int], shard_index: int = 0, num_shards: int = 1) -> None:
    if not ids:
        print("Nothing to embed")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    shard_ids = ids[shard_index::num_shards]
    image_paths = [IMAGES_DIR / f"{pid}.jpg" for pid in shard_ids]
    print(f"Shard {shard_index}/{num_shards}: {len(shard_ids)} images on {device}")

    dataset = ImageDataset(image_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=list)

    all_embeddings = []
    for images in tqdm(loader):
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings)
    write_embeddings(shard_ids, embeddings)


if __name__ == "__main__":
    ids = get_ids_to_embed(recreate=True)
    delete_embeddings_table()
    embed_images(ids)
