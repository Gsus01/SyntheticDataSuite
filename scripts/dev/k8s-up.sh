#!/usr/bin/env bash
set -euo pipefail

# Simple, verbose dev helper to bring up MinIO and/or Argo on minikube
# Usage examples:
#   scripts/dev/k8s-up.sh               # bring up all (minio + argo + db)
#   scripts/dev/k8s-up.sh --only minio  # only minio
#   scripts/dev/k8s-up.sh --only argo   # only argo
#   scripts/dev/k8s-up.sh --only db     # only postgres (component registry DB)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${MINIKUBE_PROFILE:-minikube}"
CTX="${K8S_CONTEXT:-minikube}"

info() { echo "[k8s-up] $*"; }
warn() { echo "[k8s-up][warn] $*" >&2; }

# Ensure PATH for non-interactive shells (e.g., make) and validate kubectl
export PATH="$PATH:/usr/local/bin:/usr/bin:/bin:/snap/bin:$HOME/.local/bin"
KUBECTL_CMD=()
if command -v kubectl >/dev/null 2>&1; then
  KUBECTL_CMD=(kubectl)
else
  if command -v minikube >/dev/null 2>&1; then
    info "kubectl no encontrado; usando 'minikube kubectl --' como fallback."
    KUBECTL_CMD=(minikube kubectl --)
  else
    warn "kubectl no está en PATH y tampoco se encontró minikube. Añade su ruta al PATH o instala kubectl en /usr/local/bin."
    exit 1
  fi
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

wait_for_default_service_account() {
  local namespace="$1"
  local attempts=0

  info "Esperando a que Kubernetes cree el service account por defecto en $namespace..."
  until kc -n "$namespace" get serviceaccount default >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if (( attempts >= 30 )); then
      warn "No apareció serviceaccount/default en $namespace dentro del tiempo esperado."
      exit 1
    fi
    sleep 1
  done
}

deploy_db() {
  info "Aplicando manifests de PostgreSQL..."
  kc apply -f "$ROOT_DIR/deploy/k8s/namespace.yaml"
  wait_for_default_service_account syntheticdata
  kc apply -f "$ROOT_DIR/deploy/postgres/postgres.yaml"

  info "Esperando a que PostgreSQL esté listo..."
  kc -n syntheticdata rollout status statefulset/postgres --timeout=180s
}

ensure_minikube() {
  if ! command -v minikube >/dev/null 2>&1; then
    warn "minikube no está instalado. Por favor, instálalo primero."
    exit 1
  fi
  if ! minikube status --profile "$PROFILE" >/dev/null 2>&1; then
    info "Iniciando minikube (perfil=$PROFILE)..."
    minikube start --profile "$PROFILE" --cpus=4 --memory=8192
  else
    info "minikube ya está iniciado (perfil=$PROFILE)."
  fi
  kc config use-context "$CTX" >/dev/null
}

deploy_minio() {
  info "Aplicando manifests de MinIO..."
  kc apply -f "$ROOT_DIR/deploy/minio/namespace.yaml"
  wait_for_default_service_account minio-dev

  # MinIO corre como Pod suelto en dev; si cambia un campo inmutable
  # (por ejemplo, hostPath) hay que recrearlo antes de aplicar.
  kc -n minio-dev delete pod minio --ignore-not-found >/dev/null
  kc apply -f "$ROOT_DIR/deploy/minio/minio-dev.yaml"
  kc apply -f "$ROOT_DIR/deploy/minio/minio-service.yaml"

  info "Esperando a que el Pod de MinIO esté listo..."
  kc -n minio-dev wait --for=condition=Ready pod -l app=minio --timeout=180s
}

deploy_argo() {
  info "Creando namespace argo e instalando Argo Workflows..."
  kc create namespace argo --dry-run=client -o yaml | kc apply -f -
  # kc apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml
  kc apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.7.4/install.yaml

  info "Aplicando RBAC y credenciales de MinIO para Argo..."
  kc apply -f "$ROOT_DIR/deploy/argo/minio-creds-secret.yaml"
  kc apply -f "$ROOT_DIR/deploy/argo/argo-workflow-role.yaml"
  kc apply -f "$ROOT_DIR/deploy/argo/argo-workflow-rolebinding.yaml"
  kc apply -f "$ROOT_DIR/deploy/argo/artifact-repositories-configmap.yaml"
  kc apply -f "$ROOT_DIR/deploy/argo/artifact-repository-configmap.yaml"

  info "Forzando modo de autenticación 'server' en argo-server (entorno dev)..."
  local argo_args
  argo_args="$(kc -n argo get deployment argo-server -o jsonpath='{.spec.template.spec.containers[0].args[*]}' 2>/dev/null || true)"
  if [[ " $argo_args " != *" --auth-mode=server "* ]]; then
    kc -n argo patch deployment argo-server \
      --type=json \
      -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--auth-mode=server"}]'
  fi

  info "Esperando a que Argo esté listo..."
  kc -n argo rollout status deployment/argo-server
  kc -n argo rollout status deployment/workflow-controller
}

main() {
  info "Preparando entorno Kubernetes (minikube/contexto)..."
  ensure_minikube

  if should_do minio; then
    deploy_minio
  fi
  if should_do argo; then
    deploy_argo
  fi

  # Default behavior: bring up the DB as part of the full stack
  if should_do db || [[ -z "$ONLY_COMPONENT" ]]; then
    deploy_db
  fi

  if [[ -z "$ONLY_COMPONENT" ]]; then
    info "Despliegue completo (MinIO + Argo + DB) finalizado."
  else
    info "Despliegue de componente '$ONLY_COMPONENT' finalizado."
  fi
}

main "$@"
