import os
from typing import List, Tuple

class Settings:
    APP_TITLE: str = "Edge AI - Monitoramento de Zonas de Risco (NR-12)"
    APP_VERSION: str = "1.0.0"
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/yolov8n.onnx")
    CONF_THRESHOLD: float = float(os.getenv("CONF_THRESHOLD", "0.45"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    INPUT_WIDTH: int = 640
    INPUT_HEIGHT: int = 640
    PERSON_CLASS_ID: int = 0  # Classe 'person' no dataset COCO

    # Polígonos normalizados (coordenadas relativas 0.0 a 1.0 da imagem)
    # Zona Vermelha (Perigo Crítico / Exclusão Imediata - Parada de Máquina)
    DANGER_ZONE_NORM: List[Tuple[float, float]] = [
        (0.60, 0.35),
        (0.98, 0.35),
        (0.98, 0.95),
        (0.60, 0.95)
    ]

    # Zona Amarela (Atenção / Alerta Preventivo)
    WARNING_ZONE_NORM: List[Tuple[float, float]] = [
        (0.35, 0.35),
        (0.60, 0.35),
        (0.60, 0.95),
        (0.35, 0.95)
    ]

settings = Settings()
