
FROM python:3.13-slim-bookworm


RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*


RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser


WORKDIR /app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY *.py ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY models/ ./models/
COPY notebooks/ ./notebooks/


RUN chown -R appuser:appuser /app


USER appuser


ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MEMORY_LIMIT_GB=4


CMD ["python", "src/main.py"]