#!/bin/bash        ./deploy.sh 
set -e

SERVER_IP="46.225.23.33"
REMOTE_PATH="/root/Moderator"

echo "🚀 Синхронизация кода через rsync..."

rsync -avz --delete \
      --exclude '.git/' \
      --exclude 'venv/' \
      --exclude '*.tar.gz' \
      --exclude '*.log' \
      --exclude '__pycache__/' \
      ./ root@$SERVER_IP:$REMOTE_PATH/

echo "🔄 Пересборка и перезапуск контейнера на Hetzner..."
ssh root@$SERVER_IP "cd $REMOTE_PATH && docker compose down && docker compose up -d --build"

echo "🔍 Проверка отдачи метрик Prometheus..."
sleep 3
ssh root@$SERVER_IP "curl -s -i http://127.0.0.1:8000/metrics | head -n 5"

echo "✅ Деплой завершен!"