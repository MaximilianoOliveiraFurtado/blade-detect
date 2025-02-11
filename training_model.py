from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Modelo base
model.train(data="dataset/dataset.yaml", epochs=100, imgsz=960, batch=8)