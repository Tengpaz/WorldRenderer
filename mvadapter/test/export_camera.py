import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from .utils.camera import export_camera_json

def export_camera(
    mesh_path: Path,
    blender_bin: Path,
    camera_json: Path,
) -> None:
    export_camera_json(mesh_path, blender_bin, camera_json)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export camera trajectory from a .blend file using Blender",
    )
    parser.add_argument("--device", default="cuda", help="torch device, e.g., cuda or cpu")
    parser.add_argument(
        "--blender-bin",
        type=str,
        default=str(ROOT / "blender" / "blender-5.0.0-linux-x64" / "blender"),
        help="Path to Blender binary for .blend export",
    )
    parser.add_argument(
        "--camera-json",
        type=str,
        default=str(Path(__file__).resolve().parent / "camera_path.json"),
        help="Path to save/load exported camera trajectory from blend",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device set to cuda.")

    test_dir = Path(__file__).resolve().parent
    blend_path = test_dir / "town.blend"
    blender_bin = Path(args.blender_bin)
    camera_json = test_dir / "camera_path.json"

    export_camera(
        mesh_path=blend_path,
        blender_bin=blender_bin,
        camera_json=camera_json,
    )

if __name__ == "__main__":
    main()
