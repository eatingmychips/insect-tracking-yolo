import cv2
import os
from pathlib import Path

video_dir = Path("data/raw_videos")
out_root = Path("data/frames")
out_root.mkdir(parents=True, exist_ok=True)

def process_video(video_path: Path, fps_sample: float = 5.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    # Make output subdir per video
    video_out = out_root / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # sample every Nth frame to approximate fps_sample
    step = max(int(round(original_fps / fps_sample)), 1)

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # subsample frames
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # show frame and wait for key
        cv2.imshow("frame", frame)
        key = cv2.waitKey(0) & 0xFF  # wait until key press

        if key == ord("s"):  # save
            out_path = video_out / f"{video_path.stem}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1
            print(f"Saved {out_path}")
        elif key == ord("q"):  # quit current video
            break
        # else (any other key) => skip frame

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done {video_path}, saved {saved_idx} frames")

def main():
    # loop through all mp4s in the directory
    for video_path in sorted(video_dir.glob("*.avi")):
        print(f"Processing {video_path}")
        process_video(video_path, fps_sample=5.0)

if __name__ == "__main__":
    main()