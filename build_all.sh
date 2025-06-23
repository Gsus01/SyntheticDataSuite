#!/bin/bash
# build_all.sh - Construye las imágenes Docker de todos los componentes principales
# Uso: bash build_all.sh [TAG]

# Para construilas en el entorno de minikube:
# eval $(minikube docker-env)

set -e

TAG=${1:-latest}

# Construir imagen de preprocessing
echo "Construyendo imagen: preprocessing:${TAG}"
docker build -t preprocessing:${TAG} components/preprocessing

# Construir imágenes de todos los modelos en generation
for dir in components/generation/*/; do
  if [ -f "$dir/Dockerfile" ]; then
    name=$(basename "$dir")
    echo "Construyendo imagen: generation-$name:${TAG}"
    docker build -t generation-$name:${TAG} "$dir"
  fi
done

# Construir imágenes de todos los modelos en training
for dir in components/training/*/; do
  if [ -f "$dir/Dockerfile" ]; then
    name=$(basename "$dir")
    echo "Construyendo imagen: training-$name:${TAG}"
    docker build -t training-$name:${TAG} "$dir"
  fi
done

echo "\nTodas las imágenes han sido construidas con el tag: ${TAG}"