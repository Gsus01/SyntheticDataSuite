#!/usr/bin/env bash
set -euo pipefail

# Tear down dev namespaces or selectively remove components
# Usage:
#   scripts/dev/k8s-down.sh            # remove both namespaces
#   scripts/dev/k8s-down.sh --only argo
#   scripts/dev/k8s-down.sh --only minio
#   scripts/dev/k8s-down.sh --only db

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

info() { echo "[k8s-down] $*"; }

# Ensure PATH for non-interactive shells and validate kubectl
export PATH="$PATH:/usr/local/bin:/usr/bin:/bin:/snap/bin:$HOME/.local/bin"
KUBECTL_CMD=()
if command -v kubectl >/dev/null 2>&1; then
  KUBECTL_CMD=(kubectl)
elif command -v minikube >/dev/null 2>&1; then
  info "kubectl no encontrado; usando 'minikube kubectl --'."
  KUBECTL_CMD=(minikube kubectl --)
else
  echo "[k8s-down][error] kubectl no está en PATH y minikube tampoco se encontró." >&2
  exit 1
fi

kc() {
  "${KUBECTL_CMD[@]}" "$@"
}

ONLY_COMPONENT=""
if [[ "${1:-}" == "--only" && -n "${2:-}" ]]; then
  ONLY_COMPONENT="$2"; shift 2 || true
fi

should_do() {
  local comp="$1"
  if [[ -z "$ONLY_COMPONENT" ]]; then return 0; fi
  [[ "$ONLY_COMPONENT" == "$comp" ]]
}

"$ROOT_DIR/scripts/dev/port-forward.sh" stop --only minio || true
"$ROOT_DIR/scripts/dev/port-forward.sh" stop --only argo  || true
"$ROOT_DIR/scripts/dev/port-forward.sh" stop --only db   || true

if should_do argo; then
  info "Eliminando namespace argo..."
  kc delete namespace argo --ignore-not-found
fi

if should_do minio; then
  info "Eliminando namespace minio-dev..."
  kc delete namespace minio-dev --ignore-not-found
fi

if should_do db; then
  info "Eliminando PostgreSQL (syntheticdata namespace)..."
  kc delete -n syntheticdata -f "$ROOT_DIR/deploy/postgres/postgres.yaml" --ignore-not-found
fi

info "k8s-down finalizado."


