# ==============================================================================
# Stage 1: Exporter - Baixa o YOLOv8n e exporta para formato otimizado ONNX
# ==============================================================================
FROM --platform=$BUILDPLATFORM python:3.11-slim AS model-exporter

WORKDIR /export
RUN pip install --no-cache-dir ultralytics==8.2.103 onnx==1.16.2 onnxslim==0.1.34
COPY scripts/export_onnx.py .
RUN python export_onnx.py

# ==============================================================================
# Stage 2: Runtime Embarcado Enxuto (Compatível com linux/amd64 e linux/arm64)
# ==============================================================================
FROM python:3.11-slim

LABEL maintainer="Leonardo Grando"
LABEL description="Edge AI Safety Zone Monitor (NR-12) for Raspberry Pi 5"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/yolov8n.onnx \
    CONF_THRESHOLD=0.45 \
    IOU_THRESHOLD=0.45

WORKDIR /app

# Dependências mínimas de sistema exigidas pelo OpenCV Headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o modelo ONNX gerado no primeiro estágio e o código da aplicação
COPY --from=model-exporter /export/yolov8n.onnx /app/models/yolov8n.onnx
COPY app/ /app/app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
