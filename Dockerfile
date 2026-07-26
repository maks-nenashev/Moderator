FROM python:3.11-slim
WORKDIR /app

# Системные зависимости для C/C++ сборки и OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Обновляем pip и устанавливаем библиотеки
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходники, модули обучения и артефакты
COPY artifacts/ ./artifacts/
COPY train/ ./train/
COPY src/ ./src/

# Настройка путей импорта Python
ENV PYTHONPATH=/app:/app/src

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]