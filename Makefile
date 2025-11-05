.PHONY: dev k8s-up k8s-up-minio k8s-up-argo k8s-down k8s-down-minio k8s-down-argo port-forward port-forward-stop port-forward-status backend frontend workflow

dev:
	./scripts/dev/start-stack.sh

k8s-up:
	./scripts/dev/k8s-up.sh

k8s-up-minio:
	./scripts/dev/k8s-up.sh --only minio

k8s-up-argo:
	./scripts/dev/k8s-up.sh --only argo

k8s-down:
	./scripts/dev/k8s-down.sh

k8s-down-minio:
	./scripts/dev/k8s-down.sh --only minio

k8s-down-argo:
	./scripts/dev/k8s-down.sh --only argo

port-forward:
	./scripts/dev/port-forward.sh start

port-forward-stop:
	./scripts/dev/port-forward.sh stop

port-forward-status:
	./scripts/dev/port-forward.sh status

backend:
	uv sync --project backend
	uv run --project backend uvicorn main:app --reload --host 0.0.0.0 --port 8000 --app

frontend:
	cd frontend && npm run dev

workflow:
	argo submit deploy/general_workflow.yaml -n argo --log


