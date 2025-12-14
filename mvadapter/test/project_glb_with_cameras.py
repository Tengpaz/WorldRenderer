import argparse
import glob
import json
import math
import os
import sys

import bpy
from mathutils import Matrix

try:
    from projector import ensure_uv
except Exception:
    # Fallback if the script is moved away from projector.py
    def ensure_uv(obj):
        me = obj.data
        if not hasattr(me, "uv_layers"):
            return
        if len(me.uv_layers) == 0:
            bpy.context.view_layer.objects.active = obj
            for o in bpy.context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
            bpy.ops.object.mode_set(mode="OBJECT")


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Project a set of images onto a GLB using camera poses")
    parser.add_argument("--input_glb", required=True, help="Path to source GLB")
    parser.add_argument("--camera_json", required=True, help="Camera config JSON (list of frames with matrix_world, fov_deg)")
    parser.add_argument("--images", required=True, help="Directory or glob pattern for source images")
    parser.add_argument("--output_glb", required=True, help="Path for baked GLB output")
    parser.add_argument("--targets", nargs="*", default=None, help="Optional mesh object names to project onto (defaults to all meshes)")
    parser.add_argument("--uv_offset", nargs=2, type=float, default=(0.0, 0.0), help="UV offset applied to projected UVs")
    parser.add_argument("--bake_size", type=int, default=0, help="Override bake texture width/height (square). 0 uses source image size")
    parser.add_argument("--pack", action="store_true", help="Pack images into .blend before GLB export")
    return parser.parse_args(argv)


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_glb(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    bpy.ops.import_scene.gltf(filepath=path)


def _load_camera_path(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("camera_json must contain a list of frames")
    return data


def _collect_images(spec):
    if os.path.isdir(spec):
        files = [
            os.path.join(spec, fn)
            for fn in sorted(os.listdir(spec))
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"))
        ]
    else:
        files = sorted(glob.glob(spec))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"No images found from {spec}")
    return files


def _ensure_camera(scene):
    cam_data = bpy.data.cameras.new("ProjCameraData")
    cam_obj = bpy.data.objects.new("ProjCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def _set_camera_pose(cam_obj, frame_info):
    mw = Matrix(frame_info["matrix_world"])
    cam_obj.matrix_world = mw
    cam_data = cam_obj.data
    # Optional detailed camera params if present
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
        # Fall back to single fov field captured from Blender
        cam_data.angle = math.radians(float(frame_info.get("fov_deg", 60.0)))


def _write_projected_uv(target_obj, camera_obj, uv_name, uv_offset=(0.0, 0.0)):
    from bpy_extras.object_utils import world_to_camera_view

    mesh = target_obj.data
    if uv_name not in mesh.uv_layers:
        mesh.uv_layers.new(name=uv_name)
    uv_layer = mesh.uv_layers[uv_name]
    scene = bpy.context.scene
    verts_world = [target_obj.matrix_world @ v.co for v in mesh.vertices]
    for loop in mesh.loops:
        world_co = verts_world[loop.vertex_index]
        co_ndc = world_to_camera_view(scene, camera_obj, world_co)
        u = float(co_ndc.x) + float(uv_offset[0])
        v = 1.0 - float(co_ndc.y) + float(uv_offset[1])
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)
        uv_layer.data[loop.index].uv = (u, v)
    mesh.uv_layers.active = uv_layer


def _bake_view_onto_obj(obj, cam_obj, src_img_path, bake_img, uv_offset=(0.0, 0.0), use_clear=False):
    src_img = bpy.data.images.load(src_img_path, check_existing=True)
    ensure_uv(obj)
    uv_name = "ProjectedUV"
    _write_projected_uv(obj, cam_obj, uv_name, uv_offset=uv_offset)

    mat = bpy.data.materials.new(name=f"ProjBake_{obj.name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    uvmap_node = nodes.new("ShaderNodeUVMap")
    uvmap_node.uv_map = uv_name
    uvmap_node.location = (-600, 0)

    src_tex = nodes.new("ShaderNodeTexImage")
    src_tex.image = src_img
    src_tex.extension = "CLIP"
    src_tex.interpolation = "Linear"
    src_tex.location = (-300, 0)

    emit = nodes.new("ShaderNodeEmission")
    emit.location = (0, 0)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)

    target_node = nodes.new("ShaderNodeTexImage")
    target_node.image = bake_img
    target_node.location = (-300, -200)
    nt.nodes.active = target_node

    links.new(uvmap_node.outputs["UV"], src_tex.inputs["Vector"])
    links.new(src_tex.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    scene = bpy.context.scene
    prev_engine = scene.render.engine
    scene.render.engine = "CYCLES"
    cycles = scene.cycles
    prev_bake_type = getattr(cycles, "bake_type", None)
    cycles.bake_type = "EMIT"

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", use_clear=use_clear)

    scene.render.engine = prev_engine
    if prev_bake_type is not None:
        cycles.bake_type = prev_bake_type


def _assign_baked_material(obj, bake_img):
    mat = bpy.data.materials.new(name=f"Baked_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bake_img
    tex_node.location = (-300, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def main():
    args = _parse_args()
    cam_frames = _load_camera_path(args.camera_json)
    images = _collect_images(args.images)
    if len(images) < len(cam_frames):
        print(f"Warning: fewer images ({len(images)}) than camera frames ({len(cam_frames)}); extra frames will be ignored")
    if len(images) > len(cam_frames):
        print(f"Warning: extra images ({len(images)}); they will be ignored")
    pairs = list(zip(cam_frames, images))
    _reset_scene()
    _import_glb(args.input_glb)

    scene = bpy.context.scene
    cam_obj = _ensure_camera(scene)

    mesh_objs = [o for o in scene.objects if o.type == "MESH"] if not args.targets else [scene.objects[name] for name in args.targets if name in scene.objects]
    if not mesh_objs:
        raise RuntimeError("No mesh objects to project onto")

    sample_img = bpy.data.images.load(pairs[0][1], check_existing=True)
    width, height = sample_img.size
    if args.bake_size > 0:
        width = height = int(args.bake_size)

    bake_images = {
        obj.name: bpy.data.images.new(name=f"bake_{obj.name}", width=width, height=height, alpha=True)
        for obj in mesh_objs
    }

    scene.render.resolution_x = width
    scene.render.resolution_y = height

    for idx, (frame_info, img_path) in enumerate(pairs, start=1):
        print(f"Baking view {idx}: {img_path}")
        _set_camera_pose(cam_obj, frame_info)
        for obj in mesh_objs:
            _bake_view_onto_obj(obj, cam_obj, img_path, bake_images[obj.name], uv_offset=tuple(args.uv_offset), use_clear=False)

    for obj in mesh_objs:
        _assign_baked_material(obj, bake_images[obj.name])
        if args.pack:
            bake_images[obj.name].pack()

    if args.pack:
        bpy.ops.file.pack_all()

    bpy.ops.export_scene.gltf(
        filepath=args.output_glb,
        export_format="GLB",
        use_selection=False,
        export_apply=True,
    )
    print(f"Exported baked GLB to {args.output_glb}")


if __name__ == "__main__":
    main()
