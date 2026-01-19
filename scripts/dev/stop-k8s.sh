#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# stop-k8s.sh - Stop SyntheticDataSuite in Kubernetes
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

info() { echo "[stop] $*"; }

# Handle kubectl alias
if command -v kubectl &>/dev/null; then
  KUBECTL="kubectl"
else
  KUBECTL="minikube kubectl --"
fi

info "Eliminando recursos de syntheticdata..."
$KUBECTL delete -f "$ROOT_DIR/deploy/k8s/" 2>/dev/null || true
$KUBECTL delete -n syntheticdata -f "$ROOT_DIR/deploy/postgres/postgres.yaml" 2>/dev/null || true

info ""
info "✓ Aplicación detenida"
info ""
info "MinIO y Argo siguen corriendo. Para detenerlos:"
info "  make k8s-down"
info ""
info "Para detener minikube completamente:"
info "  minikube stop"
