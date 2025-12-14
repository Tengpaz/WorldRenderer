import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Render stills from camera poses onto a scene with GLB assets")
    p.add_argument("--glb", required=True, help="GLB file providing meshes/materials/textures")
    p.add_argument("--camera_json", required=True, help="Camera config JSON (list of frames with matrix_world, fov_deg)")
    p.add_argument("--output_dir", required=True, help="Directory to write rendered images")
    p.add_argument("--output_prefix", default="render_", help="Filename prefix for outputs")
    p.add_argument("--resolution", nargs=2, type=int, metavar=("W", "H"), help="Override render resolution")
    p.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"], default="CYCLES")
    p.add_argument("--cycles_samples", type=int, default=64, help="Cycles samples if engine is CYCLES")
    p.add_argument("--clear_scene", action="store_true", help="Remove existing objects before importing GLB")
    p.add_argument("--sun", action="store_true", help="Add a white sun light to ensure lit renders")
    p.add_argument("--sun_strength", type=float, default=3.0, help="Sun light strength if --sun is set")
    p.add_argument("--world_color", nargs=3, type=float, metavar=("R", "G", "B"), help="Optional world ambient color (0-1) to avoid black scene")
    return p.parse_args(argv)


def _load_camera_frames(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("camera_json must be a list of frame dicts")
    return data


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.lights, bpy.data.cameras):
        for datablock in list(block):
            if datablock.users == 0:
                block.remove(datablock)


def _ensure_camera(scene):
    if scene.camera:
        return scene.camera
    cam_data = bpy.data.cameras.new("RenderCamData")
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def _set_camera_pose(cam_obj, frame_info):
    mw = Matrix(frame_info["matrix_world"])
    cam_obj.matrix_world = mw
    cam_data = cam_obj.data
    # Apply detailed camera params if present
    sensor_fit = frame_info.get("sensor_fit")
    if sensor_fit in {"AUTO", "HORIZONTAL", "VERTICAL"}:
        cam_data.sensor_fit = sensor_fit
    if "sensor_width" in frame_info:
        try:
            cam_data.sensor_width = float(frame_info["sensor_width"])
        except Exception:
            pass
    if "sensor_height" in frame_info:
        try:
            cam_data.sensor_height = float(frame_info["sensor_height"])
        except Exception:
            pass
    if "shift_x" in frame_info:
        try:
            cam_data.shift_x = float(frame_info["shift_x"])
        except Exception:
            pass
    if "shift_y" in frame_info:
        try:
            cam_data.shift_y = float(frame_info["shift_y"])
        except Exception:
            pass
    if "lens" in frame_info:
        try:
            cam_data.lens = float(frame_info["lens"])
        except Exception:
            pass
    else:
        fov_deg = float(frame_info.get("fov_deg", 60.0))
        cam_data.angle = math.radians(fov_deg)


def _import_glb(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    bpy.ops.import_scene.gltf(filepath=path)


def _configure_render(scene, engine, samples=None, resolution=None):
    scene.render.engine = engine
    if resolution:
        scene.render.resolution_x, scene.render.resolution_y = resolution
    if engine == "CYCLES" and samples:
        scene.cycles.samples = samples
    # Ensure we render color PNGs
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    # Disable compositor override (prevents Z/Mist-only outputs from custom node trees)
    scene.use_nodes = False
    # Color management to standard to avoid unexpected transforms
    scene.view_settings.view_transform = "Standard"


def _ensure_lighting(scene, add_sun=False, sun_strength=3.0, world_color=None):
    if add_sun:
        sun_data = bpy.data.lights.new(name="AutoSun", type="SUN")
        sun_data.energy = sun_strength
        sun_obj = bpy.data.objects.new(name="AutoSun", object_data=sun_data)
        scene.collection.objects.link(sun_obj)
        sun_obj.rotation_euler = (0.5, 0.5, 0.5)
    if world_color:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (*world_color, 1.0)


def _render_frames(cam_obj, frames, output_dir, prefix):
    scene = bpy.context.scene
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame_info in enumerate(frames, start=1):
        _set_camera_pose(cam_obj, frame_info)
        scene.render.filepath = os.path.join(output_dir, f"{prefix}{idx:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {scene.render.filepath}")


def main():
    args = _parse_args()
    frames = _load_camera_frames(args.camera_json)
    if args.clear_scene:
        _clear_scene()
    _import_glb(args.glb)
    scene = bpy.context.scene
    cam_obj = _ensure_camera(scene)
    _configure_render(scene, args.engine, samples=args.cycles_samples, resolution=args.resolution)
    _ensure_lighting(scene, add_sun=args.sun, sun_strength=args.sun_strength, world_color=args.world_color)
    _render_frames(cam_obj, frames, args.output_dir, args.output_prefix)


if __name__ == "__main__":
    main()
