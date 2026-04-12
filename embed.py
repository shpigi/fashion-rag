from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"
BATCH_SIZE = 32


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    ids = np.array([int(p.stem) for p in image_paths])
    print(f"Encoding {len(ids)} images on {device}")

    all_embeddings = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings)
    np.savez(EMBEDDINGS_FILE, ids=ids, embeddings=embeddings)
    print(f"Saved {len(ids)} embeddings to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
