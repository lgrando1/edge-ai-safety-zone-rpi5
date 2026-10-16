# ==============================================================================
# Stage 1: Exporter - Roda nativamente na arquitetura rapida do host (amd64)
# ==============================================================================
FROM --platform=$BUILDPLATFORM python:3.11-slim AS model-exporter

WORKDIR /export

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Fixa torch 2.4.1 CPU (compativel com torch.load do ultralytics) + onnxruntime
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    "numpy<2.0.0" \
    ultralytics==8.3.20 \
    onnx==1.17.0 \
    onnxruntime==1.19.2

COPY scripts/export_onnx.py .
RUN python export_onnx.py

# ==============================================================================
# Stage 2: Runtime Embarcado Enxuto (Compilado para linux/amd64 e linux/arm64)
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

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=model-exporter /export/yolov8n.onnx /app/models/yolov8n.onnx
COPY app/ /app/app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
