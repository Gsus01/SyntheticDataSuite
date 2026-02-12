#!/usr/bin/env bash
set -euo pipefail

# Orchestrate dev stack: k8s (minio/argo/db), port-forwards, backend, frontend
# Flags:
#   --skip-k8s       no k8s setup
#   --skip-pf        no port-forwards
#   --skip-backend   no backend
#   --skip-frontend  no frontend
#   --only-k8s       only k8s (implies skipping others)
#   --only-backend   only backend
#   --only-frontend  only frontend

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="$ROOT_DIR/.tmp/dev"
STACK_PID_FILE="$STATE_DIR/stack-local.pids"
LOCK_FILE="$STATE_DIR/start-stack.lock"
mkdir -p "$STATE_DIR"

DO_K8S=1
DO_PF=1
DO_BACKEND=1
DO_FRONTEND=1

PIDS=()
CLEANING_UP=0

info() { echo "[stack] $*"; }
warn() { echo "[stack][warn] $*" >&2; }

is_stack_process() {
  local pid="$1"
  local cmd
  cmd="$(ps -o command= -p "$pid" 2>/dev/null || true)"
  [[ -n "$cmd" ]] || return 1
  [[ "$cmd" == *"$ROOT_DIR/backend"* || "$cmd" == *"$ROOT_DIR/frontend"* || "$cmd" == *"uvicorn main:app"* || "$cmd" == *"next dev"* ]]
}

kill_process_group() {
  local pid="$1"
  local signal="${2:-TERM}"
  local pgid=""

  if [[ -z "$pid" ]] || ! ps -p "$pid" >/dev/null 2>&1; then
    return
  fi

  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "$pgid" ]]; then
    kill "-$signal" "-$pgid" 2>/dev/null || true
  else
    kill "-$signal" "$pid" 2>/dev/null || true
  fi
}

cleanup_stale_stack_pids() {
  if [[ ! -f "$STACK_PID_FILE" ]]; then
    return
  fi

  info "Limpiando procesos locales de una ejecución previa..."
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    if is_stack_process "$pid"; then
      kill_process_group "$pid" TERM
    fi
  done < "$STACK_PID_FILE"

  sleep 1

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    if is_stack_process "$pid"; then
      kill_process_group "$pid" KILL
    fi
  done < "$STACK_PID_FILE"

  rm -f "$STACK_PID_FILE"
}

cleanup_orphan_stack_processes() {
  local pid
  local orphan_pids=()

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    orphan_pids+=("$pid")
  done < <(
    ps -eo pid=,ppid=,command= | awk -v root="$ROOT_DIR" '
      $2 == 1 && index($0, root) &&
      (index($0, "next dev") || index($0, "uvicorn main:app") || index($0, "npm --prefix")) { print $1 }
    '
  )

  if [[ ${#orphan_pids[@]} -eq 0 ]]; then
    return
  fi

  warn "Se encontraron procesos huérfanos del stack local: ${orphan_pids[*]}"
  for pid in "${orphan_pids[@]}"; do
    kill_process_group "$pid" TERM
  done
  sleep 1
  for pid in "${orphan_pids[@]}"; do
    kill_process_group "$pid" KILL
  done
}

register_pid() {
  local pid="$1"
  PIDS+=("$pid")
  echo "$pid" >> "$STACK_PID_FILE"
}

acquire_instance_lock() {
  if ! command -v flock >/dev/null 2>&1; then
    warn "'flock' no está disponible; se omite bloqueo de instancia."
    return
  fi

  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[stack][error] Ya hay otra instancia de start-stack.sh ejecutándose." >&2
    echo "[stack][error] Cierra la sesión anterior o mata ese proceso antes de relanzar make dev." >&2
    exit 1
  fi
}

start_in_process_group() {
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" &
  else
    "$@" &
  fi
  register_pid "$!"
}

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

acquire_instance_lock
cleanup_stale_stack_pids
cleanup_orphan_stack_processes
: > "$STACK_PID_FILE"

cleanup() {
  if [[ "$CLEANING_UP" -eq 1 ]]; then
    return
  fi
  CLEANING_UP=1

  info "Cerrando procesos..."
  for pid in "${PIDS[@]}"; do
    kill_process_group "$pid" TERM
  done

  sleep 1

  for pid in "${PIDS[@]}"; do
    kill_process_group "$pid" KILL
  done

  rm -f "$STACK_PID_FILE"

  if [[ "$DO_PF" -eq 1 ]]; then
    "$ROOT_DIR/scripts/dev/port-forward.sh" stop || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$DO_K8S" -eq 1 ]]; then
  info "Levantando Kubernetes (MinIO/Argo/DB)..."
  "$ROOT_DIR/scripts/dev/k8s-up.sh"
fi

if [[ "$DO_PF" -eq 1 ]]; then
  info "Abriendo port-forwards (MinIO/Argo/DB)..."
  "$ROOT_DIR/scripts/dev/port-forward.sh" start
fi

if [[ "$DO_BACKEND" -eq 1 ]]; then
  info "Arrancando backend (http://localhost:8000)..."

  # Defaults de MinIO en dev si no están definidos
  export MINIO_ENDPOINT="${MINIO_ENDPOINT:-localhost:9000}"
  export MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
  export MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"
  export MINIO_SECURE="${MINIO_SECURE:-0}"

  # Puerto local para Postgres en dev (port-forward). Default: 5432.
  DB_LOCAL_PORT="${DB_LOCAL_PORT:-5432}"
  if ! [[ "$DB_LOCAL_PORT" =~ ^[0-9]+$ ]] || (( DB_LOCAL_PORT < 1 || DB_LOCAL_PORT > 65535 )); then
    echo "[stack][error] DB_LOCAL_PORT inválido: '$DB_LOCAL_PORT' (usa 1-65535)." >&2
    exit 1
  fi

  # DB (Postgres) defaults for local dev via port-forward
  export DATABASE_URL="${DATABASE_URL:-postgresql+psycopg://syntheticdata:syntheticdata@localhost:${DB_LOCAL_PORT}/syntheticdata}"
  export RUN_MIGRATIONS="${RUN_MIGRATIONS:-true}"

  # Sync de dependencias desde pyproject/uv.lock y lanzamiento con uv
  uv sync --project "$ROOT_DIR/backend"
  start_in_process_group uv run --project "$ROOT_DIR/backend" uvicorn main:app --reload --host 0.0.0.0 --port 8000 --app-dir "$ROOT_DIR/backend"
fi

if [[ "$DO_FRONTEND" -eq 1 ]]; then
  FRONTEND_PORT="${FRONTEND_PORT:-3000}"
  if ! [[ "$FRONTEND_PORT" =~ ^[0-9]+$ ]] || (( FRONTEND_PORT < 1 || FRONTEND_PORT > 65535 )); then
    echo "[stack][error] FRONTEND_PORT inválido: '$FRONTEND_PORT' (usa 1-65535)." >&2
    exit 1
  fi

  info "Arrancando frontend (http://localhost:${FRONTEND_PORT})..."
  # Use relative /api path - Next.js rewrites will proxy to backend
  # This allows remote access without exposing port 8000 directly
  export NEXT_PUBLIC_API_BASE_URL="/api"

  # Verificar si node_modules existe, si no, ejecutar npm install
  if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
    info "node_modules no encontrado, ejecutando npm install..."
    npm --prefix "$ROOT_DIR/frontend" install
  fi

  start_in_process_group npm --prefix "$ROOT_DIR/frontend" run dev -- --port "$FRONTEND_PORT"
fi

if [[ ${#PIDS[@]} -gt 0 ]]; then
  info "Stack en marcha. Usa Ctrl+C para parar."
  set +e
  wait -n "${PIDS[@]}"
  wait_status=$?
  set -e
  if [[ "$wait_status" -ne 0 ]]; then
    warn "Un proceso local terminó con código $wait_status."
  else
    warn "Uno de los procesos locales terminó."
  fi
else
  info "No hay procesos locales (backend/frontend) activos. Manteniendo script para señales..."
  # Mantener el script vivo para que el trap pueda cerrar PF si es necesario
  while true; do sleep 3600; done
fi
