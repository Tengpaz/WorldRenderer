import argparse
import sys
import os
from pathlib import Path

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
from .utils.glb import export_blend_to_glb
from .utils.video import load_frames
from .utils.camera import export_camera_json, load_camera_from_json
from .utils.file import save_frames, save_depth_frames_16bit
from .utils.pipeline_texture import TexturePipeline, ModProcessConfig

def project_and_render(
    mesh_path: Path,
    video_path: Path,
    output_dir: Path,
    height: int,
    width: int,
    blender_bin: Path,
    device: str,
    uv_size: int,
    frame_step: int,
    max_frames: int,
    ctx_type: str,
    next_camera_json: Path,
    axis_convert: bool,
    debug: bool,
) -> None:
    ifproject = True
    if video_path is None:
        ifproject = False
    glb_path = mesh_path.with_suffix(".glb")
    camera_json = output_dir / "camera.json"
    export_blend_to_glb(mesh_path, glb_path, blender_bin)
    export_camera_json(mesh_path, blender_bin, camera_json)

    video_frame_height, video_frame_width = height, width
    num_views_all = max_frames if max_frames > 0 else 1
    if ifproject:
        frames_np = load_frames(video_path, 0, frame_step, max_frames)
        num_views_all, video_frame_height, video_frame_width = frames_np.shape[:3]

    cam_from_blend, clip_near, clip_far = load_camera_from_json(
        camera_json, video_frame_height, video_frame_width, device, max_frames, axis_convert
    )

    num_views = min(num_views_all, len(cam_from_blend))
    if ifproject:
        frames_np = frames_np[:num_views]
    cam = cam_from_blend[:num_views]

    # Use TexturePipeline with in-memory frames and camera override (Option B)
    tp = TexturePipeline(
        upscaler_ckpt_path=None,
        inpaint_ckpt_path=None,
        device=device,
        ctx_type=ctx_type,
    )
    
    if ifproject:
        if video_path.suffix.lower() in [".mp4"]: # Use in-memory frames for video input
            images_pt = torch.tensor(frames_np, device=device, dtype=torch.float32)
            tp_out = tp(
                mesh_path=str(glb_path),
                save_dir=str(output_dir),
                save_name="projected",
                move_to_center=False,
                front_x=False,
                keep_original_transform=True,
                uv_size=uv_size,
                uv_unwarp=True,
                rgb_tensor=images_pt,
                rgb_process_config=ModProcessConfig(inpaint_mode="uv"),
                camera_projection_type="CUSTOM",
                custom_camera_json=None,
                cameras_override=cam,
                debug_mode=debug,
            )
        else:
            tp_out = tp(
                mesh_path=str(glb_path),
                save_dir=str(output_dir),
                save_name="projected",
                move_to_center=False,
                front_x=False,
                keep_original_transform=True,
                uv_size=uv_size,
                uv_unwarp=True,
                rgb_path=str(video_path),
                rgb_tensor=None,
                rgb_process_config=ModProcessConfig(inpaint_mode="uv"),
                camera_projection_type="CUSTOM",
                custom_camera_json=None,
                cameras_override=cam,
                debug_mode=debug,
            )

        mesh = load_mesh(
            mesh_path = tp_out.shaded_model_save_path if tp_out.shaded_model_save_path is not None else str(glb_path),
            rescale=False,
            move_to_center=False, # 设置是否将模型移动到中心
            front_x_to_y=False, # 设置是否将模型的前方向从X轴调整为Y轴
            default_uv_size=uv_size, # 设置默认的UV贴图大小
            merge_vertices=True, # 合并顶点以修复潜在问题
            device = device,
        )

        # 将投影得到的 UV 纹理写回到 mesh 上
        if tp_out.uv_proj_rgb is None:
            raise RuntimeError("TexturePipeline returned no RGB UV projection.")
        mesh.texture = tp_out.uv_proj_rgb.to(device)
        # 使用投影时的 UV（与 uv_proj 对齐），避免 GLB 导出丢失或不匹配导致采样全黑
        if hasattr(tp_out, "mesh_v_tex") and tp_out.mesh_v_tex is not None:
            mesh.v_tex = tp_out.mesh_v_tex.to(device)
        if hasattr(tp_out, "mesh_t_tex_idx") and tp_out.mesh_t_tex_idx is not None:
            mesh.t_tex_idx = tp_out.mesh_t_tex_idx.to(device)

        if debug:
            uv_dir = output_dir / "debug"
            uv_dir.mkdir(parents=True, exist_ok=True)
            tensor_to_image(tp_out.uv_proj_rgb).save(uv_dir / "uv_proj.png")
    else:
        try:
            from mvadapter.utils.mesh_utils.mesh_process import process_raw
        except Exception as e:
            raise ImportError(
                "uv_unwarp 请求但是 mvadapter.utils.mesh_utils.mesh_process (pymeshlab) 不可用. "
                "安装 pymeshlab 或禁用 uv_unwarp."
            ) from e

        mesh_path = str(glb_path)
        file_suffix = os.path.splitext(mesh_path)[-1]
        mesh_path_new = mesh_path.replace(file_suffix, f"_unwarp{file_suffix}")
        process_raw(mesh_path, mesh_path_new, preprocess=False)
        mesh_path = mesh_path_new
        mesh = load_mesh(
            mesh_path = mesh_path,
            rescale=False,
            move_to_center=False, # 设置是否将模型移动到中心
            front_x_to_y=False, # 设置是否将模型的前方向从X轴调整为Y轴
            default_uv_size=uv_size, # 设置默认的UV贴图大小
            merge_vertices=True, # 合并顶点以修复潜在问题
            device = device,
        )

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

    # 单位为mm的深度归一化
    norm_range = 65.535
    depth_norm = SimpleNormalization(
        scale=1.0 / norm_range,
        offset=0.0,
        clamp=True, # 超过范围的深度值会被截断
        bg_value=1.0, # 背景深度值设为1.0（即最大深度）
    )

    rgb_frames, depth_frames, normal_frames, mask_frames = [], [], [], []
    # 如果提供了 next_camera_json，则使用其中的相机路径进行渲染
    if next_camera_json != None and next_camera_json.exists():
        cam_from_blend, _, _ = load_camera_from_json(
            next_camera_json, video_frame_height, video_frame_width, device, num_views_all, axis_convert
        )
        cam = cam_from_blend[:num_views]
    # 使用相机路径渲染每一帧
    for i in range(num_views):
        cam_i = cam[i]
        render_out = render(
            ctx,
            mesh,
            cam_i,
            height=height if height is not None else video_frame_height,
            width=width if width is not None else video_frame_width,
            render_attr=True,
            render_depth=True,
            render_normal=True,
            depth_normalization_strategy=depth_norm,
            attr_background=0.0,
        )
        rgb = render_out.attr[0]
        geo_mask = render_out.mask[0]
        # Mask only where texture was actually projected (nonzero RGB), not all geometry
        tex_mask = (rgb.abs().sum(-1) > 1e-6) & geo_mask
        rgb = torch.where(tex_mask[..., None], rgb, torch.zeros_like(rgb))
        # rgb = torch.where(geo_mask, rgb, torch.zeros_like(rgb))
        depth = torch.where(geo_mask, render_out.depth[0], torch.ones_like(render_out.depth[0]))
        normal = render_out.normal[0]
        rgb_frames.append(rgb.cpu())
        depth_frames.append(depth.cpu())
        normal_frames.append(normal.cpu())
        mask_frames.append(tex_mask.cpu())

    if ifproject:
        save_frames(rgb_frames, output_dir / "rgb", "rgb")
        save_frames(mask_frames, output_dir / "mask", "mask")
    save_depth_frames_16bit(depth_frames, output_dir / "depth", "depth")
    save_frames(normal_frames, output_dir / "normal", "normal")

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
        "--debug",
        action="store_true",
        help="Dump uv_proj/uv_mask/input frame for debugging projection",
    )
    parser.add_argument(
        "--axis-convert",
        action="store_true",
        help="Apply Blender->glTF axis conversion to camera matrices (enable only if views misaligned)",
    )
    parser.add_argument(
        "--blend-path",
        type=str,
        help="Path to the .blend file to process (overrides default town.blend)",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        help="Path to the input video file (overrides default town.mp4)",
    )
    parser.add_argument(
        "--next-camera-json",
        type=str,
        help="Path to the next camera trajectory json for rendering (optional)",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Height of the rendered frames",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Width of the rendered frames",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device set to cuda.")

    test_dir = Path(__file__).resolve().parent
    blend_path = Path(args.blend_path) if args.blend_path else test_dir / "town.blend"
    video_path = Path(args.video_path) if args.video_path else None
    output_dir = Path(args.output_dir)
    blender_bin = Path(args.blender_bin)
    next_camera_json = Path(args.next_camera_json) if args.next_camera_json else None

    project_and_render(
        mesh_path=blend_path,
        video_path=video_path,
        output_dir=output_dir,
        height=args.height,
        width=args.width,
        blender_bin=blender_bin,
        device=device,
        uv_size=args.uv_size,
        frame_step=max(1, args.frame_step),
        max_frames=args.max_frames,
        ctx_type=args.ctx_type,
        next_camera_json=next_camera_json, # 如果使用下一个相机路径渲染，则设置 next_camera_json
        axis_convert=args.axis_convert,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
