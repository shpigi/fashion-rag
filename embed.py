import numpy as np
import torch
from google.cloud import bigquery
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from upath import UPath

GCP_PROJECT = "fashion-rag"
BQ_DATASET = "fashion"
BQ_EMBEDDINGS_TABLE = f"{GCP_PROJECT}.{BQ_DATASET}.clip_embeddings"
BUCKET = "gs://fashion-data-500"
IMAGES_DIR = UPath(BUCKET) / "images"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
NUM_WORKERS = 8

BQ_SCHEMA = [
    bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
]


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with self.paths[idx].open("rb") as f:
            return Image.open(f).convert("RGB")


def get_image_ids():
    client = bigquery.Client(project=GCP_PROJECT)
    rows = client.query(f"SELECT id FROM `{BQ_DATASET}.metadata` ORDER BY id").result()
    return [row.id for row in rows]


def write_embeddings(ids, embeddings):
    client = bigquery.Client(project=GCP_PROJECT)
    rows = [
        {"id": int(pid), "embedding": emb.tolist()}
        for pid, emb in zip(ids, embeddings)
    ]
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=BQ_SCHEMA,
    )
    job = client.load_table_from_json(rows, BQ_EMBEDDINGS_TABLE, job_config=job_config)
    job.result()
    print(f"Wrote {len(rows)} rows to {BQ_EMBEDDINGS_TABLE}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    ids = get_image_ids()
    image_paths = [IMAGES_DIR / f"{pid}.jpg" for pid in ids]
    print(f"Encoding {len(ids)} images from {IMAGES_DIR} on {device}")

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
    write_embeddings(ids, embeddings)


if __name__ == "__main__":
    main()
