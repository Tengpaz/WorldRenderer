import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from spandrel import ModelLoader

# Ensure project root on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from mvadapter.utils import image_to_tensor, make_image_grid, tensor_to_image
from mvadapter.utils.mesh_utils import (
    Camera,
    CameraProjection,
    NVDiffRastContextWrapper,
    SmartPainter,
    TexturedMesh,
    get_camera,
    get_orthogonal_camera,
    load_mesh,
    render,
    replace_mesh_texture_and_save,
)
# from mvadapter.utils.mesh_utils.mesh_process import process_raw

def clear():
    torch.cuda.empty_cache()

@contextmanager
def mesh_use_texture(mesh: TexturedMesh, texture: torch.FloatTensor):
    texture_ = mesh.texture
    mesh.texture = texture
    try:
        yield
    finally:
        mesh.texture = texture_


@dataclass
class ModProcessConfig:
    view_upscale: bool = False
    view_upscale_factor: int = 2
    inpaint_mode: str = "uv"  # in ["none", "uv", "view"]
    view_inpaint_max_view_score_thresh: float = 0.02
    view_inpaint_min_rounds: int = 4
    view_inpaint_max_rounds: int = 8
    view_inpaint_uv_padding_end: bool = True

@dataclass
class TexturePipelineOutput:
    shaded_model_save_path: Optional[str] = None
    pbr_model_save_path: Optional[str] = None
    uv_proj_rgb: Optional[torch.FloatTensor] = None

class TexturePipeline:
    def __init__(
        self,
        upscaler_ckpt_path: str,
        inpaint_ckpt_path: str,
        device: str,
        ctx_type: str = "cuda",  # use cuda context to avoid EGL on headless
    ):
        self.device = device
        self.ctx = NVDiffRastContextWrapper(device=self.device, context_type=ctx_type)
        self.cam_proj = CameraProjection(
            pb_backend="torch-cuda",
            bg_remover=None,
            device=self.device,
            context_type=ctx_type
        )
        if upscaler_ckpt_path is not None:
            self.upscaler = ModelLoader().load_from_file(upscaler_ckpt_path)
            self.upscaler.to(self.device).eval().half()
        if inpaint_ckpt_path is not None:
            self.inpainter = ModelLoader().load_from_file(inpaint_ckpt_path)
            self.inpainter.to(self.device).eval()

        self.smart_painter = SmartPainter(self.device, context_type=ctx_type)

    def load_packed_images(
        self, packed_image_path: Optional[str], num_views: Optional[int] = None
    ) -> List[Image.Image]:
        """Load images for projection. Supports directory, comma-separated paths, or single packed strip.

        - If a directory is given, loads and sorts image files inside.
        - If a comma-separated list is given, loads each file in order.
        - If a single file is given:
            * If num_views is provided and width is divisible by num_views, split horizontally.
            * Otherwise, treat it as a single image.
        """
        if packed_image_path is None:
            return None

        def open_img(p):
            return Image.open(p).convert("RGB")

        if os.path.isdir(packed_image_path):
            exts = {".png", ".jpg", ".jpeg", ".webp"}
            files = sorted(
                [
                    os.path.join(packed_image_path, f)
                    for f in os.listdir(packed_image_path)
                    if os.path.splitext(f)[1].lower() in exts
                ]
            )
            images = [open_img(f) for f in files]
        elif "," in packed_image_path:
            files = [p.strip() for p in packed_image_path.split(",") if p.strip()]
            images = [open_img(f) for f in files]
        else:
            img = open_img(packed_image_path)
            images = [img]

        if num_views is not None:
            if len(images) == 1 and num_views > 1:
                w, h = images[0].size
                if w % num_views == 0:
                    split_w = w // num_views
                    images = [images[0].crop((i * split_w, 0, (i + 1) * split_w, h)) for i in range(num_views)]
                else:
                    raise ValueError(
                        f"Single image width {w} not divisible by num_views={num_views}; cannot split evenly."
                    )
            elif len(images) != num_views:
                raise ValueError(
                    f"Loaded {len(images)} images but expected {num_views}. Provide exactly one per camera or a single packed strip."
                )

        return images

    def maybe_upscale_image(
        self,
        tensor: Optional[torch.FloatTensor],
        upscale: bool,
        upscale_factor: int,
        batched: bool = False,
    ) -> Optional[torch.FloatTensor]:
        if upscale:
            with torch.no_grad():
                tensor = tensor.permute(0, 3, 1, 2)
                if batched:
                    tensor = self.upscaler(tensor.half()).float()
                else:
                    tensor = torch.concat(
                        [
                            self.upscaler(im.unsqueeze(0).half()).float()
                            for im in tensor
                        ],
                        dim=0,
                    )
                tensor = tensor.clamp(0, 1).permute(0, 2, 3, 1)
            clear()
        return tensor

    def view_inpaint(
        self,
        mod_name: str,
        mesh: TexturedMesh,
        uv_proj: torch.FloatTensor,
        uv_valid_mask: torch.BoolTensor,
        config: ModProcessConfig,
        debug_dir: Optional[str] = None,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        def inpaint_func(
            image: torch.FloatTensor, mask: torch.FloatTensor
        ) -> torch.FloatTensor:
            with torch.no_grad():
                return self.inpainter(image.to(torch.float32), mask.to(torch.float32))

        return self.smart_painter(
            mod_name,
            mesh,
            inpaint_func,
            uv_proj,
            ~uv_valid_mask,
            max_view_score_thresh=config.view_inpaint_max_view_score_thresh,
            min_rounds=config.view_inpaint_min_rounds,
            max_rounds=config.view_inpaint_max_rounds,
            uv_padding_end=config.view_inpaint_uv_padding_end,
            debug_dir=debug_dir,
            debug_visualize_details=False,
        )

    def __call__(
        self,
        mesh_path: str, # 输入的3D网格模型路径
        save_dir: str, # 输出结果保存目录
        save_name: str = "default", # 输出文件的基本名称
        # mesh load settings
        move_to_center: bool = False,
        front_x: bool = False, # 将模型的前方向从X轴调整为Y轴
        keep_original_transform: bool = True,
        # uv unwarp
        uv_unwarp: bool = False, # 是否对UV贴图进行展开处理
        preprocess_mesh: bool = False, # 是否对网格进行预处理
        # projection
        uv_size: int = 4096,
        # modes
        rgb_path: Optional[str] = None, # RGB贴图路径
        rgb_tensor: Optional[torch.FloatTensor] = None, # 直接传入的RGB张量
        rgb_process_config: ModProcessConfig = ModProcessConfig(),
        base_color_path: Optional[str] = None, # 基础颜色贴图路径
        base_color_process_config: ModProcessConfig = ModProcessConfig(),
        orm_path: Optional[str] = None, # ORM贴图路径
        orm_process_config: ModProcessConfig = ModProcessConfig(),
        normal_path: Optional[str] = None, # 法线贴图路径
        normal_strength: float = 1.0, # 法线贴图强度
        normal_process_config: ModProcessConfig = ModProcessConfig(),
        # inpaint
        uv_inpaint_use_network: bool = False, # 是否使用神经网络进行UV贴图修复
        view_inpaint_include_occlusion_boundary: bool = False, # 视图修复是否包含遮挡边界
        poisson_reprojection: bool = False, # 是否进行泊松重投影
        # camera
        camera_projection_type: str = "ORTHO", # 相机投影类型，支持 "PERSP", "ORTHO", "CUSTOM"
        custom_camera_json: Optional[str] = None, # 自定义相机参数的JSON文件路径
        cameras_override: Optional[Union[Camera, List[Camera]]] = None, # 外部传入相机，跳过构建
        camera_elevation_deg: List[float] = [0, 0, 0, 0, 89.99, -89.99],
        camera_azimuth_deg: List[float] = [0, 90, 180, 270, 180, 180],
        camera_distance: float = 1.0, # 相机距离
        camera_ortho_scale: float = 1.1, # 正交相机的缩放比例
        camera_fov_deg: float = 40, # 透视相机的视场角
        # debug
        debug_mode: bool = False,
    ):
        clear()

        if debug_mode:
            debug_dir = os.path.join(save_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)

        if uv_unwarp:
            try:
                from mvadapter.utils.mesh_utils.mesh_process import process_raw
            except Exception as e:
                raise ImportError(
                    "uv_unwarp 请求但是 mvadapter.utils.mesh_utils.mesh_process (pymeshlab) 不可用. "
                    "安装 pymeshlab 或禁用 uv_unwarp."
                ) from e

            file_suffix = os.path.splitext(mesh_path)[-1]
            mesh_path_new = mesh_path.replace(file_suffix, f"_unwarp{file_suffix}")
            process_raw(mesh_path, mesh_path_new, preprocess=preprocess_mesh)
            mesh_path = mesh_path_new

        mesh: TexturedMesh = load_mesh(
            mesh_path,
            rescale=not keep_original_transform, # 设置是否重置模型大小
            move_to_center=False if keep_original_transform else move_to_center, # 设置是否将模型移动到中心
            front_x_to_y=False if keep_original_transform else front_x, # 设置是否将模型的前方向从X轴调整为Y轴
            default_uv_size=uv_size, # 设置默认的UV贴图大小
            device=self.device,
        )

        # Ensure there is a texture tensor to blend into during projection
        if mesh.texture is None:
            mesh.texture = torch.zeros((uv_size, uv_size, 3), dtype=torch.float32, device=self.device)

        # Fallback UVs if mesh lacks UV coordinates
        if mesh.v_tex is None:
            uv_done = False
            try:
                import xatlas  # type: ignore

                v_np = mesh.v_pos.detach().cpu().numpy().astype(np.float32)
                f_np = mesh.t_pos_idx.detach().cpu().numpy().astype(np.int32)
                atlas = xatlas.Atlas()
                atlas.add_mesh(v_np, f_np)
                atlas.generate()
                chart_uvs, _, _ = atlas.get_mesh(0)
                uv = torch.from_numpy(chart_uvs[:, :2]).to(self.device)
                uv_min, _ = torch.min(uv, dim=0, keepdim=True)
                uv_max, _ = torch.max(uv, dim=0, keepdim=True)
                uv = (uv - uv_min) / (uv_max - uv_min + 1e-8)
                mesh.v_tex = uv
                mesh.t_tex_idx = mesh.t_pos_idx.clone().to(self.device)
                uv_done = True
                if debug_mode:
                    print("Fallback UV: generated with xatlas")
            except Exception as e:
                if debug_mode:
                    print(f"Fallback UV: xatlas unwrap failed ({e}), using bbox planar XY.")

            if not uv_done:
                v = mesh.v_pos
                v_min, _ = torch.min(v, dim=0, keepdim=True)
                v_max, _ = torch.max(v, dim=0, keepdim=True)
                span = (v_max - v_min).clamp(min=1e-6)
                uv_xy = (v[:, :2] - v_min[:, :2]) / span[:, :2]
                mesh.v_tex = uv_xy.to(self.device)
                mesh.t_tex_idx = mesh.t_pos_idx.clone().to(self.device)

        # projection
        cameras = None
        custom_cam_cache = None
        custom_cam_data = None
        if cameras_override is not None:
            cameras = cameras_override
            expected_views = len(cameras_override) if hasattr(cameras_override, "__len__") else None
        else:
            expected_views = 6 if camera_projection_type == "ORTHO" else None
        if cameras_override is not None:
            pass
        elif camera_projection_type == "PERSP":
            raise NotImplementedError
        elif camera_projection_type == "ORTHO":
            cameras = get_orthogonal_camera(
                elevation_deg=camera_elevation_deg,
                distance=[camera_distance] * 6,
                left=-camera_ortho_scale / 2,
                right=camera_ortho_scale / 2,
                bottom=-camera_ortho_scale / 2,
                top=camera_ortho_scale / 2,
                azimuth_deg=[x - 90 for x in camera_azimuth_deg],  # -y as front
                device=self.device,
            )
        elif camera_projection_type == "CUSTOM":
            if custom_camera_json is None:
                raise ValueError("CUSTOM camera 需要 custom_camera_json 参数.")
            import json

            with open(custom_camera_json, "r") as f:
                custom_cam_data = json.load(f)
            expected_views = len(custom_cam_data)
        else:
            raise ValueError(f"不支持的相机投影类型: {camera_projection_type}")

        mod_kwargs = {
            "rgb": (rgb_path, rgb_process_config), # rgb贴图路径和处理配置
            "base_color": (base_color_path, base_color_process_config), # 基础颜色贴图路径和处理配置
            "orm": (orm_path, orm_process_config), # ORM贴图路径和处理配置
            "normal": (normal_path, normal_process_config), # 法线贴图路径和处理配置
        }
        mod_uv_image, mod_uv_tensor = {}, {}
        for mod_name, (mod_path, mod_process_config) in mod_kwargs.items():
            if mod_path is None:
                if mod_name == "rgb" and rgb_tensor is not None:
                    # Normalize in-memory frames to [0,1] if needed
                    mod_tensor = rgb_tensor
                    if mod_tensor.dtype != torch.float32:
                        mod_tensor = mod_tensor.to(torch.float32)
                    max_val = mod_tensor.max().item()
                    if max_val > 1.0:
                        mod_tensor = (mod_tensor / 255.0).clamp(0.0, 1.0)
                else:
                    mod_uv_image[mod_name] = None
                    mod_uv_tensor[mod_name] = None
                    continue
            else:
                mod_images = self.load_packed_images(mod_path, num_views=expected_views)
                mod_tensor = image_to_tensor(mod_images, device=self.device)
            mod_tensor = self.maybe_upscale_image(
                mod_tensor,
                mod_process_config.view_upscale,
                mod_process_config.view_upscale_factor,
            )
            # Build custom cameras lazily when we know view count and aspect
            if cameras_override is None and camera_projection_type == "CUSTOM" and custom_cam_cache is None:
                H, W = mod_tensor.shape[1:3]
                c2w_list, fov_list = [], []
                for item in custom_cam_data:
                    c2w_list.append(torch.tensor(item["matrix_world"], dtype=torch.float32, device=self.device))
                    fov_list.append(float(item.get("fov_deg", camera_fov_deg)))
                c2w = torch.stack(c2w_list, dim=0)
                fov = torch.tensor(fov_list, dtype=torch.float32, device=self.device)
                cameras = get_camera(c2w=c2w, fovy_deg=fov, aspect_wh=W / H, device=self.device)
                custom_cam_cache = True

            if mod_process_config.view_upscale and debug_mode:
                make_image_grid(tensor_to_image(mod_tensor, batched=True), rows=1).save(
                    os.path.join(debug_dir, f"{mod_name}_upscaled.jpg")
                )

            if mod_name == "normal":
                _, height, width, _ = mod_tensor.shape
                render_out = render(
                    self.ctx,
                    mesh,
                    cameras,
                    height=height,
                    width=width,
                    render_attr=False,
                    render_depth=False,
                    render_normal=True,
                    render_tangent=True,
                )

                # compute UV tangent space
                vN = render_out.normal
                vT = render_out.tangent
                vB = torch.cross(vN, vT, dim=-1)
                tang_space = F.normalize(torch.stack([vT, vB, vN], dim=-2), dim=-1)

                # compute geometry tangent space
                vGN = vN
                vGT = torch.as_tensor(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [-1, 0, 0],
                        [-1, 0, 0],
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )[:, None, None, :]
                vGB = torch.cross(vGN, vGT, dim=-1)
                vGT = torch.cross(vGB, vGN, dim=-1)
                geo_tang_space = F.normalize(
                    torch.stack([vGT, vGB, vGN], dim=-2), dim=-1
                )

                # restore world-space normal from geometry tangent space
                mod_tensor = mod_tensor * 2 - 1
                mod_tensor = F.normalize(
                    (
                        mod_tensor[:, :, :, None, :]
                        * geo_tang_space.permute(0, 1, 2, 4, 3)
                    ).sum(-1),
                    dim=-1,
                )

                # bake world-space normal to UV tangent space
                mod_tensor = F.normalize(
                    (mod_tensor[:, :, :, None, :] * tang_space).sum(-1), dim=-1
                )
                mod_tensor = (mod_tensor * 0.5 + 0.5).clamp(0, 1)

                view_weights = torch.ones(mod_tensor.shape[0], device=self.device)
                uv_proj, uv_valid_mask = self.cam_proj(
                    mod_tensor,
                    mesh,
                    cameras,
                    from_scratch=mod_process_config.inpaint_mode != "none",
                    poisson_blending=False,
                    depth_grad_dilation=5,
                    uv_exp_blend_alpha=3,
                    uv_exp_blend_view_weight=view_weights,
                    aoi_cos_valid_threshold=0.2,
                    uv_size=uv_size,
                    return_uv_projection_mask=True,
                )
                uv_proj[~uv_valid_mask] = torch.as_tensor([0.5, 0.5, 1]).to(uv_proj)
            else:
                # TODO: tweak depth_grad_dilation
                view_weights = torch.ones(mod_tensor.shape[0], device=self.device)
                cam_proj_out = self.cam_proj(
                    mod_tensor,
                    mesh,
                    cameras,
                    from_scratch=mod_process_config.inpaint_mode != "none",
                    poisson_blending=False,
                    depth_grad_dilation=5,
                    depth_grad_threshold=0.1,
                    uv_exp_blend_alpha=3,
                    uv_exp_blend_view_weight=view_weights,
                    aoi_cos_valid_threshold=0.2,
                    uv_size=uv_size,
                    uv_padding=not uv_inpaint_use_network,
                    return_dict=True,
                )
                uv_proj, uv_valid_mask, uv_depth_grad = (
                    cam_proj_out.uv_proj,
                    cam_proj_out.uv_proj_mask,
                    cam_proj_out.uv_depth_grad,
                )
                if uv_inpaint_use_network:
                    uv_inpaint_mask_input = 1 - uv_valid_mask[None, None].float()
                    uv_inpaint_image_input = uv_proj[None].permute(0, 3, 1, 2)
                    with torch.no_grad():
                        uv_inpaint_result = self.inpainter(
                            uv_inpaint_image_input, uv_inpaint_mask_input
                        )[0].permute(1, 2, 0)
                    clear()
                    if debug_mode:
                        make_image_grid(
                            [
                                tensor_to_image(uv_proj),
                                tensor_to_image(uv_valid_mask),
                                tensor_to_image(uv_inpaint_result),
                            ]
                        ).save(os.path.join(debug_dir, f"{mod_name}_uv_inpaint.jpg"))
                    uv_proj = uv_inpaint_result.contiguous()

                if mod_process_config.inpaint_mode == "view":
                    if view_inpaint_include_occlusion_boundary:
                        uv_max_depth_grad = uv_depth_grad.max(dim=0)[0]
                        uv_valid_mask = uv_valid_mask & (uv_max_depth_grad < 0.1)
                    uv_proj, uv_valid_mask = self.view_inpaint(
                        mod_name,
                        mesh,
                        uv_proj,
                        uv_valid_mask,
                        mod_process_config,
                        debug_dir=debug_dir if debug_mode else None,
                    )

                if poisson_reprojection:
                    # up and down
                    mesh.texture = uv_proj
                    uv_proj = self.cam_proj(
                        mod_tensor[4:5],
                        mesh,
                        cameras[4:5],
                        from_scratch=False,
                        poisson_blending=True,
                        pb_keep_original_border=True,
                        depth_grad_dilation=5,
                        uv_exp_blend_alpha=3,
                        uv_exp_blend_view_weight=torch.as_tensor([1, 1]),
                        aoi_cos_valid_threshold=0.2,
                        uv_size=uv_size,
                        uv_padding=True,
                        return_dict=False,
                    )
                    # front, sides and back
                    mesh.texture = uv_proj
                    uv_proj = self.cam_proj(
                        mod_tensor[0:4],
                        mesh,
                        cameras[0:4],
                        from_scratch=False,
                        poisson_blending=True,
                        pb_keep_original_border=True,
                        depth_grad_dilation=5,
                        uv_exp_blend_alpha=3,
                        uv_exp_blend_view_weight=torch.as_tensor([1, 1, 1, 1]),
                        aoi_cos_valid_threshold=0.2,
                        uv_size=uv_size,
                        uv_padding=True,
                        return_dict=False,
                    )

                if mod_name == "orm":
                    uv_proj[:, :, 0] = 1.0

            mod_uv_image[mod_name] = tensor_to_image(uv_proj)
            mod_uv_tensor[mod_name] = uv_proj
            clear()

        shaded_model_save_path = None
        if mod_uv_image["rgb"] is not None:
            shaded_model_save_path = os.path.join(save_dir, f"{save_name}_shaded.glb")
            replace_mesh_texture_and_save(
                mesh_path,
                shaded_model_save_path,
                texture=mod_uv_image["rgb"],
                backend="gltflib",
                task_id=save_name,
            )
        pbr_model_save_path = None
        if mod_uv_image["base_color"] is not None:
            pbr_model_save_path = os.path.join(save_dir, f"{save_name}_pbr.glb")
            replace_mesh_texture_and_save(
                mesh_path,
                pbr_model_save_path,
                texture=mod_uv_image["base_color"],
                metallic_roughness_texture=mod_uv_image["orm"],
                normal_texture=mod_uv_image["normal"],
                normal_strength=normal_strength,
                backend="gltflib",
                task_id=save_name,
            )

        clear()

        return TexturePipelineOutput(
            shaded_model_save_path=shaded_model_save_path,
            pbr_model_save_path=pbr_model_save_path,
            uv_proj_rgb=mod_uv_tensor.get("rgb"),
        )
