#!/usr/bin/env bash
set -euo pipefail

# Orchestrate dev stack: k8s (minio/argo), port-forwards, backend, frontend
# Flags:
#   --skip-k8s       no k8s setup
#   --skip-pf        no port-forwards
#   --skip-backend   no backend
#   --skip-frontend  no frontend
#   --only-k8s       only k8s (implies skipping others)
#   --only-backend   only backend
#   --only-frontend  only frontend

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DO_K8S=1
DO_PF=1
DO_BACKEND=1
DO_FRONTEND=1

info() { echo "[stack] $*"; }
warn() { echo "[stack][warn] $*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-k8s) DO_K8S=0 ;;
    --skip-pf) DO_PF=0 ;;
    --skip-backend) DO_BACKEND=0 ;;
    --skip-frontend) DO_FRONTEND=0 ;;
    --only-k8s) DO_K8S=1; DO_PF=0; DO_BACKEND=0; DO_FRONTEND=0 ;;
    --only-backend) DO_K8S=0; DO_PF=0; DO_BACKEND=1; DO_FRONTEND=0 ;;
    --only-frontend) DO_K8S=0; DO_PF=0; DO_BACKEND=0; DO_FRONTEND=1 ;;
    *) warn "Flag desconocida: $1" ;;
  esac
  shift
done

cleanup() {
  info "Cerrando procesos..."
  pkill -P $$ 2>/dev/null || true
  if [[ "$DO_PF" -eq 1 ]]; then
    "$ROOT_DIR/scripts/dev/port-forward.sh" stop || true
  fi
}
trap cleanup EXIT

if [[ "$DO_K8S" -eq 1 ]]; then
  info "Levantando Kubernetes (MinIO/Argo)..."
  "$ROOT_DIR/scripts/dev/k8s-up.sh"
fi

if [[ "$DO_PF" -eq 1 ]]; then
  info "Abriendo port-forwards (MinIO/Argo)..."
  "$ROOT_DIR/scripts/dev/port-forward.sh" start
fi

PIDS=()

if [[ "$DO_BACKEND" -eq 1 ]]; then
  info "Arrancando backend (http://localhost:8000)..."

  # Defaults de MinIO en dev si no están definidos
  export MINIO_ENDPOINT="${MINIO_ENDPOINT:-localhost:9000}"
  export MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
  export MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"
  export MINIO_SECURE="${MINIO_SECURE:-0}"

  # Sync de dependencias desde pyproject/uv.lock y lanzamiento con uv
  uv sync --project "$ROOT_DIR/backend"
  uv run --project "$ROOT_DIR/backend" uvicorn main:app --reload --host 0.0.0.0 --port 8000 --app-dir "$ROOT_DIR/backend" &
  PIDS+=($!)
fi

if [[ "$DO_FRONTEND" -eq 1 ]]; then
  info "Arrancando frontend (http://localhost:3000)..."
  # En desarrollo local, el frontend debe conectar directamente al backend local
  # (no a través de Ingress que usa /api)
  export NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
  npm --prefix "$ROOT_DIR/frontend" run dev &
  PIDS+=($!)
fi

if [[ ${#PIDS[@]} -gt 0 ]]; then
  info "Stack en marcha. Usa Ctrl+C para parar."
  wait -n "${PIDS[@]}"
else
  info "No hay procesos locales (backend/frontend) activos. Manteniendo script para señales..."
  # Mantener el script vivo para que el trap pueda cerrar PF si es necesario
  while true; do sleep 3600; done
fi


