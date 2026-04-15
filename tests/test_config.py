from fashion_rag.config import (
    BQ_DATASET,
    BQ_EMBEDDINGS_TABLE,
    BQ_METADATA_TABLE,
    BUCKET,
    COMPONENT_IMAGE,
    GCP_PROJECT,
    GCP_REGION,
    IMAGES_DIR,
    MODEL_NAME,
)


def test_bq_tables_reference_project_and_dataset():
    prefix = f"{GCP_PROJECT}.{BQ_DATASET}."
    assert BQ_METADATA_TABLE.startswith(prefix)
    assert BQ_EMBEDDINGS_TABLE.startswith(prefix)


def test_images_dir_under_bucket():
    assert str(IMAGES_DIR).startswith(BUCKET)


def test_component_image_uses_project_and_region():
    assert GCP_REGION in COMPONENT_IMAGE
    assert GCP_PROJECT in COMPONENT_IMAGE


def test_model_name_is_clip():
    assert "clip" in MODEL_NAME.lower()
