import csv
from pathlib import Path
from ultralytics import YOLO

model = YOLO("runs/detect/yolov8n_insect/weights/best.pt")
video_path = Path(r"data\raw_videos\data_2026-03-10_15-14-04.avi")
results = model(str(video_path), save=True)  # saves annotated video automatically[web:22][web:135][web:136]