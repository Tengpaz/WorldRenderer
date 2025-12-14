import numpy as np
from PIL import Image
from pathlib import Path
from mvadapter.utils.mesh_utils.utils import tensor_to_image

def save_frames(frames, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        tensor_to_image(frame).save(out_dir / f"{prefix}_{i:05d}.png")

def save_depth_frames_16bit(frames, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        arr = frame.detach().cpu().numpy()
        arr = np.clip(arr, 0.0, 1.0)
        arr16 = (arr * 65535.0 + 0.5).astype(np.uint16)
        img = Image.fromarray(arr16, mode="I;16")
        img.save(out_dir / f"{prefix}_{i:05d}.png")