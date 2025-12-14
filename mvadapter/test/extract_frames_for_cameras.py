import argparse
import json
import os
from pathlib import Path

import cv2


def read_camera_frames(cam_json_path):
    data = json.loads(Path(cam_json_path).read_text())
    frames = []
    for i, item in enumerate(data):
        # Prefer explicit frame index from export; otherwise use sequential index
        frames.append(int(item.get("frame", i)))
    return frames


def extract_frames(video_path, frame_ids, output_dir, frame_offset=-1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_dir, exist_ok=True)

    for idx, fid in enumerate(frame_ids):
        target = fid + frame_offset
        if target < 0 or target >= total:
            print(f"[warn] camera idx {idx} frame {fid} (target {target}) out of range [0,{total-1}], skip")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            print(f"[warn] failed to read frame {target} for camera idx {idx}, skip")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_path = Path(output_dir) / f"frame_{idx:05d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"saved {out_path} (cam_idx={idx}, video_frame={target})")

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Extract video frames aligned to camera_path.json frames.")
    parser.add_argument("--video", required=True, help="Input video file (e.g., video.mp4)")
    parser.add_argument("--camera-json", default="camera_path.json", help="camera json with frame indices")
    parser.add_argument("--output", default="frames", help="Directory to save extracted frames")
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=-1,
        help="Offset applied to camera frame ids before seeking video (Blender export is often 1-based, so default -1)",
    )
    args = parser.parse_args()

    frame_ids = read_camera_frames(args.camera_json)
    extract_frames(args.video, frame_ids, args.output, frame_offset=args.frame_offset)


if __name__ == "__main__":
    main()
