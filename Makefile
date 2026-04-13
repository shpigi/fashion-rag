PROJECT    := fashion-rag
REGION     := us-central1
BUCKET     := gs://fashion-data-500
LOCAL_DATA := ./data
BQ_DATASET := fashion

AR_REPO        := $(REGION)-docker.pkg.dev/$(PROJECT)/containers
APP_IMAGE      := $(AR_REPO)/fashion-rag-app:latest
COMPONENT_IMAGE := $(AR_REPO)/fashion-rag-component:latest
APP_SA         := fashion-rag-app@$(PROJECT).iam.gserviceaccount.com

.PHONY: bucket upload-images upload-csv bq-dataset bq-load upload bq \
        ar-repo docker-auth service-account cloud-run-permissions \
        setup cloud-run-setup \
        docker-app docker-push-app deploy-app \
        docker-component docker-push-component \
        pipeline-compile pipeline-submit pipeline \
        embed search app app-docker

# =============================================================================
# One-time setup
# =============================================================================

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

upload-gcs: upload-images upload-csv

ar-repo:
	gcloud artifacts repositories create containers \
		--repository-format=docker \
		--location=$(REGION) \
		--project=$(PROJECT) 2>/dev/null || true

docker-auth:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev --quiet

service-account:
	gcloud iam service-accounts create fashion-rag-app \
		--display-name="Fashion RAG App" \
		--project=$(PROJECT) 2>/dev/null || true

cloud-run-permissions: service-account
	gcloud projects add-iam-policy-binding $(PROJECT) \
		--member=serviceAccount:$(APP_SA) \
		--role=roles/bigquery.dataViewer --quiet
	gcloud projects add-iam-policy-binding $(PROJECT) \
		--member=serviceAccount:$(APP_SA) \
		--role=roles/bigquery.jobUser --quiet
	gcloud projects add-iam-policy-binding $(PROJECT) \
		--member=serviceAccount:$(APP_SA) \
		--role=roles/storage.objectViewer --quiet

setup: upload bq-load
cloud-run-setup: ar-repo docker-auth cloud-run-permissions

# =============================================================================
# Dev
# =============================================================================

embed:
	uv run python -m fashion_rag.embed

search:
	uv run python -m fashion_rag.search

app:
	uv run streamlit run app/app.py

app-docker: docker-app
	docker run --rm -p 8080:8080 -v $(HOME)/.config/gcloud:/root/.config/gcloud:ro $(APP_IMAGE)

docker-app:
	docker build -f app/Dockerfile -t $(APP_IMAGE) .

docker-component:
	docker build -f vertex/Dockerfile -t $(COMPONENT_IMAGE) .

docker-push-app: docker-app
	docker push $(APP_IMAGE)

docker-push-component: docker-component
	docker push $(COMPONENT_IMAGE)

pipeline-compile:
	uv run python -m vertex.pipelines.clip_embed

pipeline-submit:
	uv run python -m vertex.pipelines.clip_embed --submit

pipeline-append: docker-push-component
	uv run python -m vertex.pipelines.clip_embed --submit --no-recreate

pipeline: docker-push-component pipeline-submit

deploy-app: docker-push-app
	gcloud run deploy fashion-rag \
		--image $(APP_IMAGE) \
		--region $(REGION) \
		--project $(PROJECT) \
		--service-account $(APP_SA) \
		--memory 2Gi \
		--port 8080 \
		--allow-unauthenticated
