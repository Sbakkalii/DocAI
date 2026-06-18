FROM node:22-alpine AS frontend-builder
WORKDIR /build/frontend
COPY frontend/ .
RUN npm ci && npm run build

FROM python:3.11-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-builder /build/frontend/dist frontend/dist

RUN addgroup --system app && adduser --system --ingroup app app && \
    mkdir -p /app/output && chown -R app:app /app /app/output

USER app
EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=http://ollama:11434

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
