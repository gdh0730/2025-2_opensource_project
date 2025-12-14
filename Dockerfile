# Dockerfile
FROM python:3.11-slim

WORKDIR /workspace

# 시스템 패키지 (psycopg2 빌드용 등)
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델/데이터/코드 복사
COPY app ./app
COPY data ./data
COPY models ./models

ENV PYTHONPATH=/workspace

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
