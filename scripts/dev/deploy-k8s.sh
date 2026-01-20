#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# deploy-k8s.sh - Deploy SyntheticDataSuite to Kubernetes (minikube)
# =============================================================================
# Deploys:
#   - MinIO (if not already running)
#   - Argo Workflows (if not already running)
#   - SyntheticData Backend
#   - SyntheticData Frontend
#   - Ingress routing
#
# Access: http://syntheticdata.local after deployment
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

info()  { echo "[deploy] $*"; }
warn()  { echo "[deploy][warn] $*" >&2; }
error() { echo "[deploy][error] $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------
check_prerequisites() {
  info "Verificando prerequisitos..."
  
  if ! command -v minikube &>/dev/null; then
    error "minikube no encontrado. Instálalo primero."
  fi
  
  # kubectl might be an alias to 'minikube kubectl' in zsh, which doesn't work in bash scripts
  # So we check for kubectl command, and if not found, create a wrapper function
  if command -v kubectl &>/dev/null; then
    KUBECTL="kubectl"
  else
    info "kubectl no encontrado como comando, usando 'minikube kubectl --'"
    KUBECTL="minikube kubectl --"
  fi
  export KUBECTL
  
  if ! command -v docker &>/dev/null; then
    error "docker no encontrado. Instálalo primero."
  fi
  
  # Check minikube is running
  if ! minikube status &>/dev/null; then
    info "Iniciando minikube..."
    minikube start
  fi
  
  info "✓ Prerequisitos OK"
}

# -----------------------------------------------------------------------------
# Enable ingress addon
# -----------------------------------------------------------------------------
enable_ingress() {
  if minikube addons list | grep -q "ingress.*enabled"; then
    info "✓ Ingress addon ya habilitado"
  else
    info "Habilitando ingress addon..."
    minikube addons enable ingress
    info "Esperando a que el ingress controller esté listo..."
    $KUBECTL wait --namespace ingress-nginx \
      --for=condition=ready pod \
      --selector=app.kubernetes.io/component=controller \
      --timeout=120s || warn "Timeout esperando ingress, puede tardar más"
  fi
}

# -----------------------------------------------------------------------------
# Deploy MinIO and Argo (using existing scripts)
# -----------------------------------------------------------------------------
deploy_dependencies() {
  info "Desplegando MinIO, Argo y DB..."
  "$ROOT_DIR/scripts/dev/k8s-up.sh"
}

# -----------------------------------------------------------------------------
# Build Docker images
# -----------------------------------------------------------------------------
build_images() {
  info "Configurando Docker para minikube..."
  eval $(minikube docker-env)
  
  info "Construyendo imágenes (backend + frontend)..."
  SKIP_COMPONENTS=1 "$ROOT_DIR/build_all.sh"
}

# -----------------------------------------------------------------------------
# Deploy application
# -----------------------------------------------------------------------------
deploy_app() {
  info "Desplegando aplicación en Kubernetes..."
  
  # Apply twice to handle namespace creation timing
  $KUBECTL apply -f "$ROOT_DIR/deploy/k8s/" 2>/dev/null || true
  sleep 2
  $KUBECTL apply -f "$ROOT_DIR/deploy/k8s/"
  
  info "Esperando a que los pods estén listos..."
  $KUBECTL wait --namespace syntheticdata \
    --for=condition=ready pod \
    --selector=app.kubernetes.io/name=backend \
    --timeout=120s || warn "Backend tardando en arrancar"
  
  $KUBECTL wait --namespace syntheticdata \
    --for=condition=ready pod \
    --selector=app.kubernetes.io/name=frontend \
    --timeout=120s || warn "Frontend tardando en arrancar"
}

# -----------------------------------------------------------------------------
# Configure /etc/hosts
# -----------------------------------------------------------------------------
configure_hosts() {
  local minikube_ip
  minikube_ip=$(minikube ip)
  
  if grep -q "syntheticdata.local" /etc/hosts; then
    info "✓ syntheticdata.local ya configurado en /etc/hosts"
    # Update if IP changed
    local current_ip
    current_ip=$(grep "syntheticdata.local" /etc/hosts | awk '{print $1}')
    if [[ "$current_ip" != "$minikube_ip" ]]; then
      warn "IP de minikube cambió. Actualiza /etc/hosts manualmente:"
      echo "  sudo sed -i 's/$current_ip/$minikube_ip/' /etc/hosts"
    fi
  else
    info ""
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "ACCIÓN REQUERIDA: Añadir entrada DNS"
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "Ejecuta este comando:"
    info ""
    echo "  echo \"$minikube_ip syntheticdata.local\" | sudo tee -a /etc/hosts"
    info ""
  fi
}

# -----------------------------------------------------------------------------
# Start port-forwards for MinIO and Argo UIs
# -----------------------------------------------------------------------------
start_port_forwards() {
  info "Iniciando port-forwards para MinIO y Argo..."
  
  # Kill any existing port-forwards on these ports
  for port in 9090 9000 2746; do
    local pid=$(lsof -ti :$port 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
      info "Cerrando proceso existente en puerto $port (PID: $pid)"
      kill $pid 2>/dev/null || true
      sleep 1
    fi
  done
  
  # Start MinIO port-forwards (both API and Console)
  $KUBECTL port-forward -n minio-dev pod/minio 9000:9000 &>/dev/null &
  $KUBECTL port-forward -n minio-dev pod/minio 9090:9090 &>/dev/null &
  
  # Start Argo Server port-forward
  $KUBECTL port-forward -n argo svc/argo-server 2746:2746 &>/dev/null &
  
  # Give them a moment to start
  sleep 2
  
  info "✓ Port-forwards iniciados en background"
}

# -----------------------------------------------------------------------------
# Print status
# -----------------------------------------------------------------------------
print_status() {
  info ""
  info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  info "✅ DESPLIEGUE COMPLETADO"
  info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  info ""
  info "URLs de acceso:"
  info "  🌐 Frontend:    http://syntheticdata.local"
  info "  🔌 Backend API: http://syntheticdata.local/api/health"
  info "  📦 MinIO:       http://localhost:9090 (minioadmin/minioadmin)"
  info "  ⚡ Argo:        https://localhost:2746"
  info ""
  info "Estado de los pods:"
  $KUBECTL get pods -n syntheticdata
  info ""
  info "Comandos útiles:"
  info "  make k8s-stop      - Detener aplicación"
  info "  make k8s-rebuild   - Reconstruir y reiniciar"
  info "  make k8s-down      - Detener todo (incluyendo MinIO/Argo)"
  info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
  info "=== Desplegando SyntheticDataSuite en Kubernetes ==="
  
  check_prerequisites
  enable_ingress
  deploy_dependencies
  build_images
  deploy_app
  start_port_forwards
  configure_hosts
  print_status
}

main "$@"
