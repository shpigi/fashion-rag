from fashion_rag.embed import BQ_SCHEMA, ImageDataset


def test_image_dataset_len():
    ds = ImageDataset(["a", "b", "c"])
    assert len(ds) == 3


def test_image_dataset_len_empty():
    ds = ImageDataset([])
    assert len(ds) == 0


def test_bq_schema_has_id_and_embedding():
    names = {f.name for f in BQ_SCHEMA}
    assert "id" in names
    assert "embedding" in names
