# scripts/train_yolo.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="insect.yaml",
    imgsz=640,
    epochs=50,
    batch=16,
    name="yolov8n_insect",
)
