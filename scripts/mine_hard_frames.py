# scripts/mine_hard_frames.py
import cv2
from pathlib import Path
from ultralytics import YOLO

# Paths
model_path = Path("runs/detect/yolov8n_insect/weights/best.pt")
video_path = Path(r"data\raw_videos\data_2026-03-12_13-30-36.avi")  # change as needed
out_dir = Path("data/frames")
out_dir.mkdir(parents=True, exist_ok=True)

# Load model
model = YOLO(str(model_path))

# Open video (for exact frames)
cap = cv2.VideoCapture(str(video_path))

frame_idx = 0
saved = 0

# Run YOLO as a generator over video frames
for result in model(str(video_path), stream=True): 
    if frame_idx > 8400: 
        ret, frame = cap.read()
        if not ret:
            break

        boxes = result.boxes
        n_det = 0 if boxes is None else len(boxes)

        save = False
        # Case 1: no detections (model missed the insect)
        if n_det == 0:
            save = True

        if n_det > 0:
            conf = boxes.conf.cpu().numpy()

            # Case 2: lowest confidence below threshold (uncertain detection)
            if conf.min() < 0.2: 
                save = True

            # Case 3: more than 1 detection when you expect only 1 insect
            if n_det > 1:
                save = True

        if save:
            out_path = out_dir / f"hard_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

    frame_idx += 1

    if saved == 50: 
        break

cap.release()
print(f"Saved {saved} hard frames to {out_dir}")