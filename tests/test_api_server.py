from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image

from api_server.server import _format_results


def test_format_results_builds_urls():
    df = pd.DataFrame(
        {
            "id": [123],
            "score": [0.9512],
            "productDisplayName": ["Red Dress"],
            "articleType": ["Dresses"],
            "baseColour": ["Red"],
            "gender": ["Women"],
            "season": ["Summer"],
            "masterCategory": ["Apparel"],
        }
    )
    results = _format_results(df)
    assert len(results) == 1
    assert results[0]["image_url"] == "/images/123.jpg"
    assert results[0]["score"] == 0.9512
    assert results[0]["id"] == 123


def test_format_results_empty():
    df = pd.DataFrame(
        columns=[
            "id",
            "score",
            "productDisplayName",
            "articleType",
            "baseColour",
            "gender",
            "season",
            "masterCategory",
        ]
    )
    assert _format_results(df) == []


def _fake_search_df():
    return pd.DataFrame(
        {
            "id": [42],
            "score": [0.85],
            "productDisplayName": ["Blue Shirt"],
            "articleType": ["Shirts"],
            "baseColour": ["Blue"],
            "gender": ["Men"],
            "season": ["Winter"],
            "masterCategory": ["Apparel"],
        }
    )


def _make_test_image_bytes():
    img = Image.new("RGB", (64, 64), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@patch("api_server.server.bq_search", return_value=_fake_search_df())
@patch("api_server.server.encode_image", return_value=np.zeros(512))
@patch("api_server.server.encode_text", return_value=np.zeros(512))
@patch("api_server.server.load_model", return_value=(MagicMock(), MagicMock()))
@patch("api_server.server.storage")
def test_search_text_endpoint(mock_storage, mock_load, mock_enc_text, mock_enc_img, mock_search):
    from api_server.server import app

    with TestClient(app) as client:
        resp = client.get("/search/text", params={"q": "blue shirt", "k": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == 42
    mock_enc_text.assert_called_once()
    mock_search.assert_called_once()


@patch("api_server.server.bq_search", return_value=_fake_search_df())
@patch("api_server.server.encode_image", return_value=np.zeros(512))
@patch("api_server.server.encode_text", return_value=np.zeros(512))
@patch("api_server.server.load_model", return_value=(MagicMock(), MagicMock()))
@patch("api_server.server.storage")
def test_search_image_endpoint(mock_storage, mock_load, mock_enc_text, mock_enc_img, mock_search):
    from api_server.server import app

    with TestClient(app) as client:
        image_bytes = _make_test_image_bytes()
        resp = client.post("/search/image", files={"file": ("test.jpg", image_bytes, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == 42
    mock_enc_img.assert_called_once()
    mock_search.assert_called_once()
