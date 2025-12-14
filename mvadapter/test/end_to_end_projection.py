import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure repository root on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper,
    load_mesh,
    render,
)
from mvadapter.utils.mesh_utils.camera import get_camera
from mvadapter.utils.mesh_utils.utils import tensor_to_image
from pipeline_texture import TexturePipeline, ModProcessConfig


def export_blend_to_glb(blend_path: Path, glb_path: Path, blender_bin: Path) -> None:
    """Export a .blend scene to GLB using Blender in headless mode."""
    if glb_path.exists():
        return
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(blender_bin),
        "-b",
        str(blend_path),
        "--python-expr",
        (
            "import bpy; "
            "bpy.ops.export_scene.gltf(filepath=r'%s', export_format='GLB')"
        )
        % glb_path,
    ]
    subprocess.run(cmd, check=True)


def export_camera_json(blend_path: Path, blender_bin: Path, json_path: Path) -> None:
    """Export camera world matrices + fov per frame from a .blend using Blender."""
    if json_path.exists():
        return
    json_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = json_path.parent / "_export_camera_tmp.py"
    script = """
import bpy, json, math
scene = bpy.context.scene
cam = scene.camera
if cam is None:
    raise RuntimeError('No active camera in scene')
data = []
for f in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(f)
    mw = cam.matrix_world
    fov = cam.data.angle * 180.0 / math.pi
    data.append({
        'frame': int(f),
        'fov_deg': float(fov),
        'clip_start': float(cam.data.clip_start),
        'clip_end': float(cam.data.clip_end),
        'matrix_world': [[float(mw[i][j]) for j in range(4)] for i in range(4)]
    })
with open(r"JSON_PATH_PLACEHOLDER", 'w') as fp:
    json.dump(data, fp)
"""
    script = script.replace("JSON_PATH_PLACEHOLDER", str(json_path))
    script_path.write_text(script)
    cmd = [str(blender_bin), "-b", str(blend_path), "--python", str(script_path)]
    subprocess.run(cmd, check=True)


def load_video_frames(video_path: Path, frame_step: int, max_frames: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
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


def load_camera_from_json(json_path: Path, height: int, width: int, device: str, axis_convert: bool):
    data = json.loads(json_path.read_text())
    if len(data) == 0:
        raise RuntimeError("Camera json is empty.")
    c2w_list, fov_list, clip_start_list, clip_end_list = [], [], [], []
    for item in data:
        mw = torch.tensor(item["matrix_world"], dtype=torch.float32, device=device)
        c2w_list.append(mw)
        fov_list.append(item["fov_deg"])
        clip_start_list.append(item.get("clip_start", 0.1))
        clip_end_list.append(item.get("clip_end", 100.0))
    c2w = torch.stack(c2w_list, dim=0)
    if axis_convert:
        axis = torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
            device=device,
        )
        axis_inv = torch.linalg.inv(axis)
        c2w = axis @ c2w @ axis_inv
    fov = torch.tensor(fov_list, dtype=torch.float32, device=device)
    cam = get_camera(c2w=c2w, fovy_deg=fov, aspect_wh=width / height, device=device)
    clip_start = torch.tensor(clip_start_list, dtype=torch.float32, device=device)
    clip_end = torch.tensor(clip_end_list, dtype=torch.float32, device=device)
    near_val = float(torch.median(clip_start).item())
    far_val = float(torch.median(clip_end).item())
    if far_val <= near_val + 1e-6:
        near_val, far_val = 0.1, 100.0
    return cam, near_val, far_val


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


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end: export GLB from blend, project video frames, render rgb/mask/depth.",
    )
    parser.add_argument("--blend", type=str, default="town.blend", help="Input .blend scene")
    parser.add_argument("--video", type=str, default="video.mp4", help="Input video")
    parser.add_argument("--output-dir", type=str, default="output_project", help="Output directory")
    parser.add_argument("--blender-bin", type=str, default=str(ROOT / "blender" / "blender-3.1.2-linux-x64" / "blender"), help="Blender binary path")
    parser.add_argument("--camera-json", type=str, default="camera_path.json", help="Where to save/load exported cameras")
    parser.add_argument("--device", type=str, default="cuda", help="torch device")
    parser.add_argument("--ctx-type", type=str, default="cuda", choices=["gl", "cuda"], help="nvdiffrast context type")
    parser.add_argument("--uv-size", type=int, default=2048, help="UV resolution")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame from video")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to use (0 = all)")
    parser.add_argument("--pb-backend", type=str, default="torch-cuda", choices=["torch-native", "torch-cuda", "triton"], help="Poisson blending backend (pipeline projection)")
    parser.add_argument("--axis-convert", action="store_true", help="Apply Blender->glTF axis conversion if needed")
    parser.add_argument("--debug-dump-uv", action="store_true", help="Dump uv_proj/uv_mask/input view (pipeline)")
    parser.add_argument("--checker-test", action="store_true", help="Render checkerboard instead of projection")
    parser.add_argument("--uv-unwarp", action="store_true", help="Run pymeshlab unwarp inside pipeline if mesh lacks UVs")
    parser.add_argument("--preprocess-mesh", action="store_true", help="Mesh preprocess before unwarp when uv_unwarp is on")
    parser.add_argument("--keep-original-transform", action="store_true", help="Keep original mesh transform (no rescale/center) during pipeline projection")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device is cuda")

    blend_path = Path(args.blend)
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    blender_bin = Path(args.blender_bin)
    camera_json = Path(args.camera_json)

    output_dir.mkdir(parents=True, exist_ok=True)

    glb_path = output_dir / (blend_path.stem + ".glb")
    export_blend_to_glb(blend_path, glb_path, blender_bin)
    export_camera_json(blend_path, blender_bin, camera_json)

    # Extract video frames to disk for pipeline_texture
    frames_np = load_video_frames(video_path, max(1, args.frame_step), args.max_frames)
    num_views_all, height, width = frames_np.shape[:3]
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames_np):
        img = Image.fromarray((frame * 255).astype(np.uint8))
        img.save(frames_dir / f"frame_{i:05d}.png")

    # Run pipeline_texture to project frames onto mesh
    pipe = TexturePipeline(
        upscaler_ckpt_path=None,
        inpaint_ckpt_path=None,
        device=device,
        ctx_type=args.ctx_type,
    )

    pipeline_out = pipe(
        mesh_path=str(glb_path),
        save_dir=str(output_dir),
        save_name="reproj",
        move_to_center=False,
        front_x=False,
        keep_original_transform=True,
        uv_size=args.uv_size,
        uv_unwarp=args.uv_unwarp,
        preprocess_mesh=args.preprocess_mesh,
        rgb_path=str(frames_dir),
        rgb_process_config=ModProcessConfig(inpaint_mode="uv"),
        camera_projection_type="CUSTOM",
        custom_camera_json=str(camera_json),
        debug_mode=args.debug_dump_uv,
    )

    textured_glb_path = pipeline_out.shaded_model_save_path or pipeline_out.pbr_model_save_path
    if textured_glb_path is None:
        raise RuntimeError("Pipeline projection did not produce a textured GLB.")

    # Build cameras for rendering
    cam_all, clip_near, clip_far = load_camera_from_json(camera_json, height, width, device, args.axis_convert)
    num_views = min(num_views_all, len(cam_all))
    cam = cam_all[:num_views]

    mesh = load_mesh(
        str(textured_glb_path),
        rescale=False,
        move_to_center=False,
        default_uv_size=args.uv_size,
        merge_vertices=True, # merge vertices to fix potential issues
        device=device,
    )

    try:
        cam_pos = cam.c2w[:, :3, 3]
        v = mesh.v_pos[None]
        diff = v - cam_pos[:, None, :]
        dist = torch.norm(diff, dim=-1)
        min_d = dist.min().item()
        max_d = dist.max().item()
        span = max_d - min_d
        pad = span * 0.05 if span > 0 else 1.0
        clip_near = max(0.0001, min_d - pad)
        clip_far = max(clip_near + 1e-4, max_d + pad)
    except Exception:
        clip_near, clip_far = 0.1, 100.0

    ctx = NVDiffRastContextWrapper(device=device, context_type=args.ctx_type)
    scale = 1.0 / (clip_far - clip_near)
    offset = -clip_near / (clip_far - clip_near)
    depth_norm = lambda d: torch.clamp(d * scale + offset, 0.0, 1.0)

    rgb_frames, depth_frames, mask_frames = [], [], []
    for i in range(num_views):
        cam_i = cam[i]
        render_out = render(
            ctx,
            mesh,
            cam_i,
            height=height,
            width=width,
            render_attr=True,
            render_depth=True,
            render_normal=False,
            depth_normalization_strategy=None,
            attr_background=0.0,
        )
        rgb = render_out.attr[0]
        geo_mask = render_out.mask[0]
        tex_mask = (rgb.abs().sum(-1) > 1e-6) & geo_mask
        rgb = torch.where(tex_mask[..., None], rgb, torch.zeros_like(rgb))
        depth = render_out.depth[0]
        depth = torch.where(geo_mask, depth_norm(depth), torch.ones_like(depth))
        rgb_frames.append(rgb.cpu())
        depth_frames.append(depth.cpu())
        mask_frames.append(tex_mask.cpu())

    save_frames(rgb_frames, output_dir / "rgb", "rgb")
    save_depth_frames_16bit(depth_frames, output_dir / "depth", "depth")
    save_frames(mask_frames, output_dir / "mask", "mask")

    print("Done. Outputs:")
    print(f"  Textured GLB: {textured_glb}")
    print(f"  RGB:   {output_dir / 'rgb'}")
    print(f"  Mask:  {output_dir / 'mask'}")
    print(f"  Depth: {output_dir / 'depth'}")


if __name__ == "__main__":
    main()
