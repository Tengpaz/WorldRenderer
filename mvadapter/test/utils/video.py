import os
from pathlib import Path
import numpy as np

import cv2

def export_frames(video_path, output_dir, frame_offset=0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_dir, exist_ok=True)

    for frame_idx in range(total):
        target = frame_idx + frame_offset
        if target < 0 or target >= total:
            print(f"[warn] frame {frame_idx} (target {target}) out of range [0,{total-1}], skip")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            print(f"[warn] failed to read frame {target}, skip")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_path = Path(output_dir) / f"frame_{frame_idx:05d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"saved {out_path} (video_frame={target})")

    cap.release()

def load_frames(video_path, frame_offset=-1, frame_step=1, max_frames=-1) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        target = idx + frame_offset
        if target < 0 or target >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            idx += 1
            continue
        if idx % frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if 0 < max_frames <= len(frames):
                break
        idx += 1
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames loaded from video.")
    return np.stack(frames, axis=0).astype(np.float32) / 255.0