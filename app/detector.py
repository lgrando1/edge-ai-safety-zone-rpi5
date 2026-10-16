import time
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from app.config import settings


class SafetyZoneDetector:
    def __init__(self, model_path: str = settings.MODEL_PATH):
        # Configura threads para otimização em CPU quad-core Cortex-A76 (Raspberry Pi 5)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _get_scaled_polygons(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Converte coordenadas normalizadas dos polígonos para pixels da imagem atual."""
        danger_poly = np.array(
            [[int(x * width), int(y * height)] for x, y in settings.DANGER_ZONE_NORM],
            dtype=np.int32
        )
        warning_poly = np.array(
            [[int(x * width), int(y * height)] for x, y in settings.WARNING_ZONE_NORM],
            dtype=np.int32
        )
        return danger_poly, warning_poly

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Etapa 1: Pré-processamento (Redimensionamento, BGR->RGB, Normalização e NCHW)."""
        orig_h, orig_w = image.shape[:2]
        resized = cv2.resize(image, (settings.INPUT_WIDTH, settings.INPUT_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        scale_x = orig_w / settings.INPUT_WIDTH
        scale_y = orig_h / settings.INPUT_HEIGHT
        return tensor, scale_x, scale_y

    def postprocess(
        self,
        outputs: np.ndarray,
        scale_x: float,
        scale_y: float,
        orig_w: int,
        orig_h: int
    ) -> List[Dict[str, Any]]:
        """Etapa 3: Pós-processamento (Filtro de classe 'person', NMS e Teste de Polígono NR-12)."""
        predictions = np.squeeze(outputs[0]).T  # Formato [8400, 84]

        boxes = []
        confidences = []

        for pred in predictions:
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if class_id == settings.PERSON_CLASS_ID and confidence >= settings.CONF_THRESHOLD:
                cx, cy, w, h = pred[0:4]
                x1 = int((cx - w / 2) * scale_x)
                y1 = int((cy - h / 2) * scale_y)
                bw = int(w * scale_x)
                bh = int(h * scale_y)

                x1 = max(0, x1)
                y1 = max(0, y1)
                boxes.append([x1, y1, bw, bh])
                confidences.append(confidence)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, settings.CONF_THRESHOLD, settings.IOU_THRESHOLD
        )

        danger_poly, warning_poly = self._get_scaled_polygons(orig_w, orig_h)
        detections = []

        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, w, h = boxes[idx]
                conf = confidences[idx]

                # Ponto de referência do operador no solo (centro da base da bounding box)
                foot_point = (int(x + w / 2), int(y + h))

                in_danger = cv2.pointPolygonTest(danger_poly, foot_point, False) >= 0
                in_warning = cv2.pointPolygonTest(warning_poly, foot_point, False) >= 0

                if in_danger:
                    zone_status = "CRITICAL_DANGER_ZONE"
                    risk_level = "HIGH"
                    action_required = "STOP_MACHINE_RELAY_AND_SOUND_ALARM"
                elif in_warning:
                    zone_status = "WARNING_BUFFER_ZONE"
                    risk_level = "MEDIUM"
                    action_required = "VISUAL_AND_AUDIBLE_WARNING"
                else:
                    zone_status = "SAFE_ZONE"
                    risk_level = "LOW"
                    action_required = "NONE"

                detections.append({
                    "class_id": settings.PERSON_CLASS_ID,
                    "class_name": "person",
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x_min": x,
                        "y_min": y,
                        "x_max": x + w,
                        "y_max": y + h
                    },
                    "reference_foot_point": {"x": foot_point[0], "y": foot_point[1]},
                    "zone_status": zone_status,
                    "risk_level": risk_level,
                    "action_required": action_required
                })

        return detections

    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        """Executa o pipeline completo de inferência e retorna métricas e detecções."""
        orig_h, orig_w = image.shape[:2]

        t0 = time.perf_counter()
        tensor, scale_x, scale_y = self.preprocess(image)
        t_pre = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        t_inf = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        detections = self.postprocess(outputs, scale_x, scale_y, orig_w, orig_h)
        t_post = (time.perf_counter() - t2) * 1000

        total_ms = t_pre + t_inf + t_post

        # Determina o status global da cena industrial
        high_risk_count = sum(1 for d in detections if d["risk_level"] == "HIGH")
        medium_risk_count = sum(1 for d in detections if d["risk_level"] == "MEDIUM")

        if high_risk_count > 0:
            global_status = "CRITICAL_ALERT_STOP_MACHINE"
        elif medium_risk_count > 0:
            global_status = "WARNING_OPERATOR_NEAR_HAZARD"
        else:
            global_status = "NORMAL_OPERATION"

        return {
            "global_scene_status": global_status,
            "summary": {
                "total_operators_detected": len(detections),
                "operators_in_danger_zone": high_risk_count,
                "operators_in_warning_zone": medium_risk_count
            },
            "detections": detections,
            "metadata": {
                "model": "YOLOv8n (ONNX Runtime CPU/ARM64 Optimized)",
                "image_resolution": {"width": orig_w, "height": orig_h},
                "timing_ms": {
                    "preprocess_ms": round(t_pre, 2),
                    "inference_ms": round(t_inf, 2),
                    "postprocess_ms": round(t_post, 2),
                    "total_pipeline_ms": round(total_ms, 2),
                    "estimated_fps": round(1000.0 / max(total_ms, 0.1), 1)
                }
            }
        }

    def annotate_image(self, image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Desenha polígonos de risco NR-12, bounding boxes, pontos de referência e painel de status."""
        annotated = image.copy()
        orig_h, orig_w = annotated.shape[:2]
        danger_poly, warning_poly = self._get_scaled_polygons(orig_w, orig_h)

        # Camada translúcida para as zonas de risco
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [warning_poly], (0, 215, 255))  # Amarelo (BGR)
        cv2.fillPoly(overlay, [danger_poly], (0, 0, 255))     # Vermelho (BGR)
        cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0, annotated)

        # Contornos das zonas
        cv2.polylines(annotated, [warning_poly], True, (0, 215, 255), 2)
        cv2.polylines(annotated, [danger_poly], True, (0, 0, 255), 3)

        cv2.putText(
            annotated, "ZONA DE ATENCAO (NR-12)",
            (warning_poly[0][0] + 8, warning_poly[0][1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2
        )
        cv2.putText(
            annotated, "ZONA CRITICA - EXCLUSAO",
            (danger_poly[0][0] + 8, danger_poly[0][1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )

        # Desenha cada operador detectado
        for det in result["detections"]:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
            foot = (det["reference_foot_point"]["x"], det["reference_foot_point"]["y"])
            risk = det["risk_level"]

            if risk == "HIGH":
                color = (0, 0, 255)      # Vermelho
            elif risk == "MEDIUM":
                color = (0, 215, 255)    # Amarelo
            else:
                color = (0, 200, 0)      # Verde

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, foot, 6, color, -1)

            label = f"Operador ({det['confidence']:.2f}) | Risco: {risk}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 3, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

        # Banner superior de telemetria
        status = result["global_scene_status"]
        banner_color = (0, 0, 200) if "CRITICAL" in status else ((0, 140, 255) if "WARNING" in status else (30, 130, 30))
        cv2.rectangle(annotated, (0, 0), (orig_w, 42), banner_color, -1)
        telemetry_text = (
            f"STATUS: {status} | Operadores: {result['summary']['total_operators_detected']} "
            f"| Latencia: {result['metadata']['timing_ms']['total_pipeline_ms']} ms"
        )
        cv2.putText(
            annotated, telemetry_text, (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        return annotated
