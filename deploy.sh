#!/bin/bash
SERVER_IP="46.225.23.33"
REMOTE_PATH="/root/Moderator"

echo "🚀 Синхронизация кода через rsync (Оптимизировано)..."

# Убедись, что исключили всё лишнее
rsync -avz --delete \
      --exclude '.git/' \
      --exclude 'venv/' \
      --exclude '*.tar.gz' \
      --exclude '*.log' \
      --exclude '__pycache__/' \
      ./ root@$SERVER_IP:$REMOTE_PATH/

echo "🔄 Перезапуск контейнера на Hetzner..."
# Исправленная команда (добавлен пробел и полная команда)
ssh root@$SERVER_IP "cd $REMOTE_PATH && docker compose up -d"

echo "✅ Деплой завершен!"

#!/bin/bash       ./deploy.sh   # Запуск деплоя
#SERVER_IP="46.225.23.33"
#IMAGE_NAME="findway-moderator:v4"

#echo "🛠 Собираем новый образ (включаем все изменения кода)..."
#docker build -t $IMAGE_NAME .

#echo "💾 Упаковываем обновленный образ..."
#docker save $IMAGE_NAME | gzip > moderator_v4.tar.gz

#echo "📤 Отправляем образ и конфиг на Hetzner..."
#scp moderator_v4.tar.gz docker-compose.yml root@$SERVER_IP:~/Moderator/

#echo "🔄 Обновляем систему на сервере..."
#ssh root@$SERVER_IP "cd ~/Moderator && docker load < moderator_v4.tar.gz && docker compose up -d"

#echo "✅ Проект полностью обновлен!"

# chmod +x deploy.sh    # Даем права на выполнение

# ./deploy.sh   # Запуск деплоя