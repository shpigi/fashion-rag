import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from google.cloud import storage
from PIL import Image

from fashion_rag.config import GCP_PROJECT, GCS_BUCKET
from fashion_rag.search import encode_image, encode_text, load_model
from fashion_rag.search import search as bq_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, processor = load_model()
    app.state.model = model
    app.state.processor = processor
    app.state.gcs = storage.Client(project=GCP_PROJECT)
    yield


app = FastAPI(title="fashion-search", lifespan=lifespan)


def _format_results(df):
    results = []
    for _, row in df.iterrows():
        item_id = int(row["id"])
        results.append(
            {
                "id": item_id,
                "image_url": f"/images/{item_id}.jpg",
                "score": round(float(row["score"]), 4),
                "productDisplayName": row.get("productDisplayName", ""),
                "articleType": row.get("articleType", ""),
                "baseColour": row.get("baseColour", ""),
                "gender": row.get("gender", ""),
                "season": row.get("season", ""),
                "masterCategory": row.get("masterCategory", ""),
            }
        )
    return results


@app.get("/images/{image_id}.jpg")
async def get_image(image_id: int):
    """Proxy an image from GCS."""
    bucket = app.state.gcs.bucket(GCS_BUCKET)
    blob = bucket.blob(f"images/{image_id}.jpg")
    if not blob.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    data = blob.download_as_bytes()
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


@app.get("/search/text")
def search_text(q: str = Query(..., description="Text query"), k: int = Query(5, ge=1, le=50)):
    """Search the catalog by text description."""
    emb = encode_text(q, app.state.model, app.state.processor)
    df = bq_search(emb, k=k)
    return _format_results(df)


@app.post("/search/image")
def search_image(file: UploadFile = File(...), k: int = Query(5, ge=1, le=50)):
    """Search the catalog by uploading an image."""
    data = file.file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    query_emb = encode_image(image, app.state.model, app.state.processor)
    df = bq_search(query_emb, k=k)
    return _format_results(df)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
