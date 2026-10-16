from ultralytics import YOLO

def export_model():
    # Carrega o modelo pré-treinado YOLOv8 Nano (COCO dataset)
    model = YOLO("yolov8n.pt")
    # Exporta para ONNX com opset 12 e resolução 640x640 otimizada para CPU/ARM64
    model.export(format="onnx", imgsz=640, opset=12)

if __name__ == "__main__":
    export_model()
