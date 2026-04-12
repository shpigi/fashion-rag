PROJECT    := fashion-rag
REGION     := us-central1
BUCKET     := gs://fashion-data-500
LOCAL_DATA := ./data
BQ_DATASET := fashion

.PHONY: bucket upload-images upload-csv bq-dataset bq-load upload bq setup

bucket:
	gcloud storage buckets create $(BUCKET) --project=$(PROJECT) --location=$(REGION) --uniform-bucket-level-access 2>/dev/null || true

upload-images: bucket
	gcloud storage cp $(LOCAL_DATA)/images/*.jpg $(BUCKET)/images/ --project=$(PROJECT)

upload-csv: bucket
	gcloud storage cp $(LOCAL_DATA)/metadata.csv $(BUCKET)/metadata.csv --project=$(PROJECT)

bq-dataset:
	bq --project_id=$(PROJECT) mk --dataset $(BQ_DATASET) 2>/dev/null || true

bq-load: bq-dataset upload-csv
	bq --project_id=$(PROJECT) load \
		--replace \
		--source_format=CSV \
		--skip_leading_rows=1 \
		--autodetect \
		$(BQ_DATASET).metadata \
		$(BUCKET)/metadata.csv
	bq --project_id=$(PROJECT) show $(BQ_DATASET).metadata

upload: upload-images upload-csv
bq: bq-load
setup: upload bq
