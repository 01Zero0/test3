#!/bin/bash
set -e

IMAGE_NAME="iris-classifier"
IMAGE_TAG="latest"
TRIVY_CACHE_DIR="${HOME}/.cache/trivy"

echo "Сборка Docker-образа..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .

echo "Очистка перед сканированием..."
docker system prune -f

echo "Запуск контейнера в тестовом режиме..."
docker run --rm \
    --user appuser \
    --memory="512m" \
    --cpus="1" \
    $IMAGE_NAME:$IMAGE_TAG python -m pytest tests/ -v

echo "Все этапы завершены успешно!"