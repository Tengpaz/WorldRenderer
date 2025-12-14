import torch
import json

from pathlib import Path
from mvadapter.utils.mesh_utils.camera import get_camera

def build_camera(num_views: int, height: int, width: int, device: str):
    azimuth = torch.linspace(0, 360, num_views + 1, device=device)[:-1]
    elevation = torch.zeros_like(azimuth, device=device)
    distance = torch.ones_like(azimuth, device=device) * 2.5
    fovy = torch.ones_like(azimuth, device=device) * 60.0
    cam = get_camera(
        elevation_deg=elevation,
        distance=distance,
        fovy_deg=fovy,
        azimuth_deg=azimuth,
        num_views=num_views,
        aspect_wh=width / height,
        device=device,
    )
    return cam

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
    cmd = [
        str(blender_bin),
        "-b",
        str(blend_path),
        "--python",
        str(script_path),
    ]
    subprocess.run(cmd, check=True)

def load_camera_from_json(
    json_path: Path,
    height: int,
    width: int,
    device: str,
    max_views: int,
    axis_convert: bool,
):
    data = json.loads(json_path.read_text())
    if len(data) == 0:
        raise RuntimeError("Camera json is empty.")
    # Use up to max_views frames
    data = data[:max_views]
    c2w_list = []
    fov_list = []
    clip_start_list = []
    clip_end_list = []
    for item in data:
        mw = torch.tensor(item["matrix_world"], dtype=torch.float32, device=device)
        c2w_list.append(mw)
        fov_list.append(item["fov_deg"])
        clip_start_list.append(item.get("clip_start", 0.1))
        clip_end_list.append(item.get("clip_end", 100.0))
    c2w = torch.stack(c2w_list, dim=0)
    if axis_convert:
        # Optional Blender -> glTF axis conversion. Only enable if you see systematic orientation mismatch.
        axis = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        axis_inv = torch.linalg.inv(axis)
        c2w = axis @ c2w @ axis_inv
    fov = torch.tensor(fov_list, dtype=torch.float32, device=device)
    clip_start = torch.tensor(clip_start_list, dtype=torch.float32, device=device)
    clip_end = torch.tensor(clip_end_list, dtype=torch.float32, device=device)
    cam = get_camera(
        c2w=c2w,
        fovy_deg=fov,
        aspect_wh=width / height,
        device=device,
    )
    # Use median near/far across frames for stable depth normalization
    near_val = float(torch.median(clip_start).item())
    far_val = float(torch.median(clip_end).item())
    if far_val <= near_val + 1e-6:
        near_val, far_val = 0.1, 100.0
    return cam, near_val, far_val