PROJECT    := fashion-rag
REGION     := us-central1
BUCKET     := gs://fashion-data-500
LOCAL_DATA := ./data
BQ_DATASET := fashion

AR_REPO        := $(REGION)-docker.pkg.dev/$(PROJECT)/containers
APP_IMAGE      := $(AR_REPO)/fashion-rag-app:latest
COMPONENT_IMAGE := $(AR_REPO)/fashion-rag-component:latest
API_IMAGE      := $(AR_REPO)/fashion-rag-api:latest
APP_SA         := fashion-rag-app@$(PROJECT).iam.gserviceaccount.com
API_PORT       ?= 8080

.PHONY: bucket upload-images upload-csv bq-dataset bq-load upload bq \
        ar-repo docker-auth service-account cloud-run-permissions \
        setup cloud-run-setup \
        docker-app docker-push-app deploy-app \
        docker-component docker-push-component \
        pipeline-compile pipeline-submit pipeline \
        docker-api docker-push-api deploy-api api-docker \
        embed search app app-docker api serve

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

api:
	uv run --extra api python -m uvicorn api_server.server:app --host 0.0.0.0 --port $(API_PORT)

app:
	FASHION_API_URL=http://localhost:$(API_PORT) uv run --extra app python -m streamlit run app/app.py

serve:
	uv run --extra api python -m uvicorn api_server.server:app --host 0.0.0.0 --port $(API_PORT) & API_PID=$$!; \
	trap "kill $$API_PID 2>/dev/null" EXIT; \
	until curl -sf http://localhost:$(API_PORT)/metadata/values >/dev/null 2>&1; do sleep 1; done; \
	FASHION_API_URL=http://localhost:$(API_PORT) uv run --extra app python -m streamlit run app/app.py

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
	$(eval API_URL := $(shell gcloud run services describe fashion-rag-api --region=$(REGION) --project=$(PROJECT) --format='value(status.url)'))
	@if [ -z "$(API_URL)" ]; then echo "Error: deploy the API first (make deploy-api)" >&2; exit 1; fi
	gcloud run deploy fashion-rag \
		--image $(APP_IMAGE) \
		--region $(REGION) \
		--project $(PROJECT) \
		--service-account $(APP_SA) \
		--memory 2Gi \
		--port 8080 \
		--set-env-vars FASHION_API_URL=$(API_URL) \
		--allow-unauthenticated

docker-api:
	docker build -f api_server/Dockerfile -t $(API_IMAGE) .

docker-push-api: docker-api
	docker push $(API_IMAGE)

api-docker: docker-api
	docker run --rm -p 8080:8080 -v $(HOME)/.config/gcloud:/root/.config/gcloud:ro $(API_IMAGE)

deploy-api: docker-push-api
	gcloud run deploy fashion-rag-api \
		--image $(API_IMAGE) \
		--region $(REGION) \
		--project $(PROJECT) \
		--service-account $(APP_SA) \
		--memory 2Gi \
		--port 8080 \
		--allow-unauthenticated
