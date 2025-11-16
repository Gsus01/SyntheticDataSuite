#!/usr/bin/env bash
set -euo pipefail

# Manage port-forward sessions for MinIO and Argo with simple status tracking
# Usage:
#   scripts/dev/port-forward.sh start [--only minio|argo]
#   scripts/dev/port-forward.sh stop  [--only minio|argo]
#   scripts/dev/port-forward.sh status

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="$ROOT_DIR/.tmp/dev"
mkdir -p "$STATE_DIR"

info() { echo "[pf] $*"; }
warn() { echo "[pf][warn] $*" >&2; }

# Ensure PATH for non-interactive shells and validate kubectl
export PATH="$PATH:/usr/local/bin:/usr/bin:/bin:/snap/bin:$HOME/.local/bin"
KUBECTL_CMD=()
if command -v kubectl >/dev/null 2>&1; then
  KUBECTL_CMD=(kubectl)
elif command -v minikube >/dev/null 2>&1; then
  info "kubectl no encontrado; usando 'minikube kubectl --'."
  KUBECTL_CMD=(minikube kubectl --)
else
  echo "[pf][error] kubectl no está en PATH y minikube tampoco se encontró." >&2
  exit 1
fi

kc() {
  "${KUBECTL_CMD[@]}" "$@"
}

ONLY_COMPONENT=""
ACTION="${1:-status}"
shift || true
if [[ "${1:-}" == "--only" && -n "${2:-}" ]]; then
  ONLY_COMPONENT="$2"; shift 2 || true
fi

should_do() {
  local comp="$1"
  if [[ -z "$ONLY_COMPONENT" ]]; then return 0; fi
  [[ "$ONLY_COMPONENT" == "$comp" ]]
}

start_pf() {
  local name="$1" namespace="$2" resource="$3" ports="$4"
  local pidf="$STATE_DIR/$name.pid" logf="$STATE_DIR/$name.log"
  if [[ -f "$pidf" ]] && ps -p "$(cat "$pidf" 2>/dev/null)" >/dev/null 2>&1; then
    info "$name ya está port-forwarding (PID $(cat "$pidf"))"
    return
  fi
  info "Iniciando port-forward de $name ($ports)..."
  set +e
  kc -n "$namespace" port-forward "$resource" $ports >"$logf" 2>&1 &
  local pid=$!
  set -e
  echo "$pid" >"$pidf"
  sleep 0.3
  if ! ps -p "$pid" >/dev/null 2>&1; then
    warn "No se pudo iniciar port-forward para $name. Revisa $logf"
    rm -f "$pidf"
  fi
}

stop_pf() {
  local name="$1"
  local pidf="$STATE_DIR/$name.pid"
  if [[ -f "$pidf" ]]; then
    local pid
    pid="$(cat "$pidf" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
      info "Deteniendo port-forward de $name (PID $pid)..."
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidf"
  else
    info "$name no estaba activo."
  fi
}

status_pf() {
  for name in minio argo; do
    local pidf="$STATE_DIR/$name.pid"
    if [[ -f "$pidf" ]] && ps -p "$(cat "$pidf")" >/dev/null 2>&1; then
      echo "$name: running (PID $(cat "$pidf"))"
    else
      echo "$name: stopped"
    fi
  done
}

case "$ACTION" in
  start)
    if should_do minio; then start_pf minio minio-dev pod/minio "9000:9000 9090:9090"; fi
    if should_do argo;  then start_pf argo  argo      deployment/argo-server "2746:2746"; fi
    ;;
  stop)
    if should_do minio; then stop_pf minio; fi
    if should_do argo;  then stop_pf argo;  fi
    ;;
  status)
    status_pf
    ;;
  *)
    echo "Uso: $0 {start|stop|status} [--only minio|argo]" >&2
    exit 1
    ;;
esac


