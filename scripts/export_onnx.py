from ultralytics import YOLO

def export_model():
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=640, opset=12)

if __name__ == "__main__":
    export_model()
