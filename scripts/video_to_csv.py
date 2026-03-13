# scripts/video_to_csv.py
import csv
from pathlib import Path
from ultralytics import YOLO

model_path = Path("runs/detect/yolov8n_insect/weights/best.pt")
video_path = Path("data\raw_videos\data_2026-03-10_15-14-04.avi")
csv_path = Path("data/data_2026-03-10_15-14-04.csv")

model = YOLO(str(model_path))

results_gen = model(str(video_path), stream=True)  # yields one result per frame[web:22][web:25][web:30]

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "class_id",
        "class_name",
        "confidence",
        "cx",
        "cy",
        "x1",
        "y1",
        "x2",
        "y2",
    ])

    frame_idx = 0
    for result in results_gen:
        boxes = result.boxes
        names = result.names

        if boxes is None or len(boxes) == 0:
            frame_idx += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        # If you know there is at most one insect, you can just take argmax here.
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            c = int(c)
            class_name = names.get(c, str(c))

            writer.writerow([
                frame_idx,
                c,
                class_name,
                float(p),
                float(cx),
                float(cy),
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ])

        frame_idx += 1

print(f"Saved detections to {csv_path}")