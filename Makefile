.PHONY: dev k8s-up k8s-up-minio k8s-up-argo k8s-down k8s-down-minio k8s-down-argo port-forward port-forward-stop port-forward-status backend frontend workflow k8s-deploy k8s-stop k8s-rebuild

# =============================================================================
# Development (local, without containers)
# =============================================================================
dev:
	./scripts/dev/start-stack.sh

backend:
	uv sync --project backend
	uv run --project backend uvicorn main:app --reload --host 0.0.0.0 --port 8000 --app

frontend:
	cd frontend && NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev

# =============================================================================
# Kubernetes Deployment (containerized)
# =============================================================================
k8s-deploy:
	./scripts/dev/deploy-k8s.sh

k8s-stop:
	./scripts/dev/stop-k8s.sh

k8s-rebuild:
	@echo "[rebuild] Reconstruyendo imágenes y reiniciando..."
	@eval $$(minikube docker-env) && SKIP_COMPONENTS=1 ./build_all.sh
	@kubectl rollout restart deployment -n syntheticdata
	@echo "[rebuild] ✓ Listo. Accede a http://syntheticdata.local"

# =============================================================================
# Infrastructure (MinIO, Argo)
# =============================================================================
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

# =============================================================================
# Port forwarding (for MinIO/Argo access)
# =============================================================================
port-forward:
	./scripts/dev/port-forward.sh start

port-forward-stop:
	./scripts/dev/port-forward.sh stop

port-forward-status:
	./scripts/dev/port-forward.sh status

# =============================================================================
# Misc
# =============================================================================
workflow:
	argo submit deploy/general_workflow.yaml -n argo --log
