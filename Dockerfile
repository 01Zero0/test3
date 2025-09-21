# Базовый образ
FROM python:3.13-slim-bookworm

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя без root-привилегий
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Создание рабочей директории
WORKDIR /app

# Копирование файлов требований
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY *.py ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY models/ ./models/
COPY notebooks/ ./notebooks/

# Смена владельца файлов
RUN chown -R appuser:appuser /app

# Переключение на непривилегированного пользователя
USER appuser

# Установка переменных окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MEMORY_LIMIT_GB=4

# Точка входа - ИСПРАВЛЕНО!
CMD ["python", "src/main.py"]