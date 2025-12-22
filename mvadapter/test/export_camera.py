import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from mvadapter.utils.mesh_utils.mesh import load_mesh
from mvadapter.utils.mesh_utils.render import (
    SimpleNormalization,
    NVDiffRastContextWrapper,
    render,
)
from mvadapter.utils.mesh_utils.utils import tensor_to_image
from .glb import export_blend_to_glb
from .video import load_frames
from .camera import build_camera, export_camera_json, load_camera_from_json
from .file import save_frames, save_depth_frames_16bit
from .pipeline_texture import TexturePipeline, ModProcessConfig

def project_and_render(
    mesh_path: Path,
    video_path: Path,
    output_dir: Path,
    blender_bin: Path,
    device: str,
    uv_size: int,
    frame_step: int,
    max_frames: int,
    pb_backend: str,
    ctx_type: str,
    camera_json: Path,
    axis_convert: bool,
    debug: bool,
) -> None:
    export_camera_json(mesh_path, blender_bin, camera_json)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Project video onto town.blend and export rgb/depth/mask frames.",
    )
    parser.add_argument("--device", default="cuda", help="torch device, e.g., cuda or cpu")
    parser.add_argument("--uv-size", type=int, default=2048, help="UV texture resolution")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame from video")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to use (0=all)")
    parser.add_argument(
        "--ctx-type",
        type=str,
        default="cuda",
        choices=["gl", "cuda"],
        help="nvdiffrast context type; use cuda if EGL/GL is unavailable",
    )
    parser.add_argument(
        "--pb-backend",
        type=str,
        default="torch-native",
        choices=["torch-native", "torch-cuda", "triton"],
        help="Poisson blending backend; 'torch-native' is pure PyTorch (no ninja).",
    )
    parser.add_argument(
        "--blender-bin",
        type=str,
        default=str(ROOT / "blender" / "blender-5.0.0-linux-x64" / "blender"),
        help="Path to Blender binary for .blend export",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output_project"),
        help="Directory to save rgb/depth/mask frames (stays inside test dir)",
    )
    parser.add_argument(
        "--camera-json",
        type=str,
        default=str(Path(__file__).resolve().parent / "camera_path.json"),
        help="Path to save/load exported camera trajectory from blend",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump uv_proj/uv_mask/input frame for debugging projection",
    )
    parser.add_argument(
        "--axis-convert",
        action="store_true",
        help="Apply Blender->glTF axis conversion to camera matrices (enable only if views misaligned)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device set to cuda.")

    test_dir = Path(__file__).resolve().parent
    blend_path = test_dir / "new_town.blend"
    video_path = test_dir / "town.mp4"
    output_dir = Path(args.output_dir)
    blender_bin = Path(args.blender_bin)
    camera_json = test_dir / "next_camera_path.json"

    project_and_render(
        mesh_path=blend_path,
        video_path=video_path,
        output_dir=output_dir,
        blender_bin=blender_bin,
        device=device,
        uv_size=args.uv_size,
        frame_step=max(1, args.frame_step),
        max_frames=args.max_frames,
        pb_backend=args.pb_backend,
        ctx_type=args.ctx_type,
        camera_json=camera_json,
        axis_convert=args.axis_convert,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
