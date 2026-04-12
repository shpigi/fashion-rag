from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from google.cloud import bigquery
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from upath import UPath

GCP_PROJECT = "fashion-rag"
BQ_DATASET = "fashion"
BUCKET = "gs://fashion-data-500"
IMAGES_DIR = UPath(BUCKET) / "images"
EMBEDDINGS_FILE = Path("data/embeddings.npz")

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


def get_image_ids():
    client = bigquery.Client(project=GCP_PROJECT)
    rows = client.query(f"SELECT id FROM `{BQ_DATASET}.metadata` ORDER BY id").result()
    return [row.id for row in rows]


def load_image(path):
    with path.open("rb") as f:
        return Image.open(f).convert("RGB")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    ids = get_image_ids()
    image_paths = [IMAGES_DIR / f"{pid}.jpg" for pid in ids]
    print(f"Encoding {len(ids)} images from {IMAGES_DIR} on {device}")

    all_embeddings = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=8) as pool:
            images = list(pool.map(load_image, batch_paths))
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings)
    np.savez(EMBEDDINGS_FILE, ids=np.array(ids), embeddings=embeddings)
    print(f"Saved {len(ids)} embeddings to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
