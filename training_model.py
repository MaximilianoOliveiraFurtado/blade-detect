from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Modelo base
model.train(data="dataset/dataset.yaml", epochs=150, imgsz=960, batch=8, freeze=10, lr0=0.0001, augment=True)
