import csv
from pathlib import Path
from ultralytics import YOLO

model = YOLO("runs/detect/yolov8n_insect3/weights/best.pt")
video_path = Path(r"data\raw_videos\data_2026-03-11_10-25-34.avi")
results = model(str(video_path), save=True)  