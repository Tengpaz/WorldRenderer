import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from mvadapter.utils.mesh_utils.mesh import load_mesh
from mvadapter.utils.mesh_utils.projection import CameraProjection
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
    debug_dump_uv: bool,
    checker_test: bool,
) -> None:
    glb_path = mesh_path.with_suffix(".glb")
    export_blend_to_glb(mesh_path, glb_path, blender_bin)
    export_camera_json(mesh_path, blender_bin, camera_json)

    frames_np = load_frames(video_path, -1, frame_step, max_frames)
    num_views_all, height, width = frames_np.shape[:3]

    cam_from_blend, clip_near, clip_far = load_camera_from_json(
        camera_json, height, width, device, num_views_all, axis_convert
    )
    num_views = min(num_views_all, len(cam_from_blend))
    frames_np = frames_np[:num_views]
    cam = cam_from_blend[:num_views]

    # Preserve original Blender world scale to keep depth consistent with camera trajectory
    mesh = load_mesh(
        str(glb_path),
        rescale=False,
        move_to_center=False,
        default_uv_size=uv_size,
        merge_vertices=True,
        device=device,
    )
    # Fallback UVs if export lost UV coordinates
    if mesh.v_tex is None:
        uv_done = False
        # 1) Try xatlas for a more accurate automatic unwrap
        try:
            import xatlas  # type: ignore

            v_np = mesh.v_pos.detach().cpu().numpy().astype(np.float32)
            f_np = mesh.t_pos_idx.detach().cpu().numpy().astype(np.int32)
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            atlas.generate()
            chart_uvs, _, _ = atlas.get_mesh(0)
            uv = torch.from_numpy(chart_uvs[:, :2]).to(device)
            # Normalize to [0,1]
            uv_min, _ = torch.min(uv, dim=0, keepdim=True)
            uv_max, _ = torch.max(uv, dim=0, keepdim=True)
            uv = (uv - uv_min) / (uv_max - uv_min + 1e-8)
            mesh.v_tex = uv
            mesh.t_tex_idx = mesh.t_pos_idx.clone().to(device)
            uv_done = True
            print("Fallback UV: generated with xatlas")
        except Exception as e:
            print(f"Fallback UV: xatlas unwrap failed ({e}), using bbox planar XY.")

        # 2) Simple planar XY fallback if xatlas unavailable
        if not uv_done:
            v = mesh.v_pos
            v_min, _ = torch.min(v, dim=0, keepdim=True)
            v_max, _ = torch.max(v, dim=0, keepdim=True)
            span = (v_max - v_min).clamp(min=1e-6)
            uv_xy = (v[:, :2] - v_min[:, :2]) / span[:, :2]
            mesh.v_tex = uv_xy.to(device)
            mesh.t_tex_idx = mesh.t_pos_idx.clone().to(device)

    if checker_test:
        # Simple black/white checkerboard to validate UV orientation/coverage
        step = uv_size // 16 if uv_size >= 16 else 1
        grid_y = torch.arange(uv_size, device=device) // step
        grid_x = torch.arange(uv_size, device=device) // step
        checker = ((grid_y[:, None] + grid_x[None]) % 2).float()
        checker = checker.unsqueeze(-1).repeat(1, 1, 3)
        mesh.texture = checker
    else:
        mesh.texture = torch.zeros((uv_size, uv_size, 3), dtype=torch.float32, device=device)

        cam_proj = CameraProjection(
            pb_backend=pb_backend,
            bg_remover=None,
            device=device,
            context_type=ctx_type,
        )
        images_pt = torch.tensor(frames_np, device=device, dtype=torch.float32)
        proj_out = cam_proj(
            images=images_pt,
            mesh=mesh,
            cam=cam,
            uv_size=uv_size,
            from_scratch=True,
            poisson_blending=False,
            uv_padding=True,
            # Relax validity filters to avoid fully-masked UVs
            aoi_cos_valid_threshold=-1.0,
            depth_grad_threshold=None,
            iou_rejection_threshold=None,
            return_dict=True,
        )
        if proj_out is None:
            raise RuntimeError("Camera projection returned None (likely IoU rejection).")
        mesh.texture = proj_out.uv_proj

        if debug_dump_uv:
            uv_dir = output_dir / "uv_debug"
            uv_dir.mkdir(parents=True, exist_ok=True)
            tensor_to_image(proj_out.uv_proj).save(uv_dir / "uv_proj.png")
            tensor_to_image(proj_out.uv_proj_mask).save(uv_dir / "uv_mask.png")
            tensor_to_image(images_pt[0]).save(uv_dir / "input_view0.png")
            print(f"uv_proj mean: {proj_out.uv_proj.mean().item():.6f}, mask mean: {proj_out.uv_proj_mask.float().mean().item():.6f}")

    # Recompute a tighter, stable depth range from camera positions and mesh bbox (similar to blender_export_depth)
    try:
        cam_pos = cam.c2w[:, :3, 3]
        v = mesh.v_pos[None]  # [1,N,3]
        diff = v - cam_pos[:, None, :]
        dist = torch.norm(diff, dim=-1)
        min_d = dist.min().item()
        max_d = dist.max().item()
        span = max_d - min_d
        pad = span * 0.05 if span > 0 else 1.0
        near = max(0.0001, min_d - pad)
        far = max(near + 1e-4, max_d + pad)
        clip_near, clip_far = near, far
    except Exception:
        pass

    ctx = NVDiffRastContextWrapper(device=device, context_type=ctx_type)
    depth_norm = SimpleNormalization(
        scale=1.0 / (clip_far - clip_near),
        offset=-clip_near / (clip_far - clip_near),
        clamp=True,
        bg_value=1.0,
    )

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
            depth_normalization_strategy=depth_norm,
            attr_background=0.0,
        )
        rgb = render_out.attr[0]
        geo_mask = render_out.mask[0]
        # Mask only where texture was actually projected (nonzero RGB), not all geometry
        tex_mask = (rgb.abs().sum(-1) > 1e-6) & geo_mask
        rgb = torch.where(tex_mask[..., None], rgb, torch.zeros_like(rgb))
        depth = torch.where(geo_mask, render_out.depth[0], torch.ones_like(render_out.depth[0]))
        rgb_frames.append(rgb.cpu())
        depth_frames.append(depth.cpu())
        mask_frames.append(tex_mask.cpu())

    save_frames(rgb_frames, output_dir / "rgb", "rgb")
    save_depth_frames_16bit(depth_frames, output_dir / "depth", "depth")
    save_frames(mask_frames, output_dir / "mask", "mask")

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
        "--debug-dump-uv",
        action="store_true",
        help="Dump uv_proj/uv_mask/input frame for debugging projection",
    )
    parser.add_argument(
        "--axis-convert",
        action="store_true",
        help="Apply Blender->glTF axis conversion to camera matrices (enable only if views misaligned)",
    )
    parser.add_argument(
        "--checker-test",
        action="store_true",
        help="Skip projection and render a checkerboard texture to verify UVs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device set to cuda.")

    test_dir = Path(__file__).resolve().parent
    blend_path = test_dir / "town.blend"
    video_path = test_dir / "video.mp4"
    output_dir = Path(args.output_dir)
    blender_bin = Path(args.blender_bin)
    camera_json = Path(args.camera_json)

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
        debug_dump_uv=args.debug_dump_uv,
        checker_test=args.checker_test,
    )


if __name__ == "__main__":
    main()
