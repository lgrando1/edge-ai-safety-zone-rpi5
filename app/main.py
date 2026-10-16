import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from app.config import settings
from app.detector import SafetyZoneDetector

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "Sistema embarcado de Visão Computacional (Edge AI) para Raspberry Pi 5 "
        "destinado ao monitoramento de comportamento de risco em zonas de máquinas pesadas (NR-12)."
    )
)

detector = SafetyZoneDetector()


def _read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Arquivo enviado não é uma imagem válida (use JPG, PNG ou WEBP)."
        )
    return image


@app.get("/health", tags=["Monitoramento"])
async def health_check():
    """Verifica a disponibilidade do serviço embarcado e metadados do modelo."""
    return {
        "status": "healthy",
        "service": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "model_path": settings.MODEL_PATH
    }


@app.post("/predict/json", tags=["Inferência Edge AI"])
async def predict_json(file: UploadFile = File(...)):
    """
    Endpoint 1 (Obrigatório): Recebe uma imagem e retorna JSON estruturado com
    detecções, classes, confiança, nível de risco NR-12 e metadados de latência.
    """
    try:
        contents = await file.read()
        image = _read_image_from_upload(contents)
        result = detector.infer(image)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no pipeline de inferência: {str(e)}")


@app.post(
    "/predict/image",
    tags=["Inferência Edge AI"],
    responses={200: {"content": {"image/png": {}}}}
)
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint 2 (Obrigatório): Recebe uma imagem e retorna um arquivo PNG anotado
    com os polígonos de risco (Zona Amarela e Vermelha), bounding boxes e telemetria.
    """
    try:
        contents = await file.read()
        image = _read_image_from_upload(contents)
        result = detector.infer(image)
        annotated = detector.annotate_image(image, result)

        success, encoded_png = cv2.imencode(".png", annotated)
        if not success:
            raise RuntimeError("Falha ao codificar a imagem anotada em PNG.")

        return Response(content=encoded_png.tobytes(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar imagem anotada: {str(e)}")
