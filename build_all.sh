#!/bin/bash
# build_all.sh - Construye las imágenes Docker de todos los componentes principales
# Uso: bash build_all.sh [TAG]
#
# Por defecto construye en el Docker de minikube. Para construir en Docker local:
#   SKIP_MINIKUBE_ENV=1 ./build_all.sh
#
# Variables de entorno opcionales:
#   SKIP_MINIKUBE_ENV=1     - No configurar Docker de minikube
#   SKIP_COMPONENTS=1       - No construir componentes ML (solo backend/frontend)
#   BACKEND_URL=...         - URL del backend para el frontend
#
# BACKEND_URL options:
#   /api                    - Para uso con Ingress (default, recomendado)
#   http://localhost:8000   - Para desarrollo local sin Ingress (port-forward)

set -e

TAG=${1:-latest}
# Default: /api for Ingress routing (frontend and backend behind same domain)
BACKEND_URL=${BACKEND_URL:-"/api"}

# Configurar el entorno Docker de minikube (a menos que se indique lo contrario)
if [ -z "$SKIP_MINIKUBE_ENV" ]; then
  if command -v minikube &> /dev/null; then
    echo "Configurando entorno Docker de minikube..."
    eval $(minikube docker-env)
  else
    echo "WARN: minikube no encontrado. Usando Docker local."
  fi
fi

# =============================================================================
# Backend y Frontend
# =============================================================================

echo ""
echo "=========================================="
echo "Construyendo Backend y Frontend"
echo "=========================================="

# Construir backend
echo "Construyendo imagen: syntheticdata-backend:${TAG}"
docker build -t syntheticdata-backend:${TAG} backend/

# Construir frontend con la URL del backend
echo "Construyendo imagen: syntheticdata-frontend:${TAG}"
echo "  NEXT_PUBLIC_API_BASE_URL=${BACKEND_URL}"
docker build -t syntheticdata-frontend:${TAG} \
  --build-arg NEXT_PUBLIC_API_BASE_URL="${BACKEND_URL}" \
  frontend/

# =============================================================================
# Componentes ML (opcional)
# =============================================================================

if [ -n "$SKIP_COMPONENTS" ]; then
  echo ""
  echo "SKIP_COMPONENTS=1 - Saltando construcción de componentes ML"
  echo ""
  echo "Backend y Frontend construidos con tag: ${TAG}"
  exit 0
fi

echo ""
echo "=========================================="
echo "Construyendo Componentes ML"
echo "=========================================="

# Construir imagen de preprocessing
echo "Construyendo imagen: preprocessing:${TAG}"
docker build -t preprocessing:${TAG} components/preprocessing

# Construir imágenes de todos los modelos en generation
for dir in components/generation/*/; do
  if [ -f "$dir/Dockerfile" ]; then
  name=$(basename "$dir")
  image_name=$(echo "$name" | tr '[:upper:]' '[:lower:]')
  echo "Construyendo imagen: generation-$image_name:${TAG}"
  docker build -t generation-$image_name:${TAG} "$dir"
  fi
done

# Construir imágenes de todos los modelos en training
for dir in components/training/*/; do
  if [ -f "$dir/Dockerfile" ]; then
  name=$(basename "$dir")
  image_name=$(echo "$name" | tr '[:upper:]' '[:lower:]')
  echo "Construyendo imagen: training-$image_name:${TAG}"
  docker build -t training-$image_name:${TAG} "$dir"
  fi
done

# Construir imágenes de RL
for dir in components/rl/*/; do
  if [ -f "$dir/Dockerfile" ]; then
    name=$(basename "$dir")
    echo "Construyendo imagen: rl-$name:${TAG}"
    docker build -t rl-$name:${TAG} "$dir"
  fi
done

# Construir imágenes de Unity/simulaciones
for dir in components/unity/*/; do
  if [ -f "$dir/Dockerfile" ]; then
    name=$(basename "$dir")
    image_name=$(echo "$name" | sed 's/\([a-z0-9]\)\([A-Z]\)/\1-\2/g' | tr '[:upper:]' '[:lower:]')
    echo "Construyendo imagen: ${image_name}:${TAG}"
    docker build -t ${image_name}:${TAG} "$dir"
  fi
done

echo ""
echo "=========================================="
echo "Todas las imágenes construidas con tag: ${TAG}"
echo "=========================================="

