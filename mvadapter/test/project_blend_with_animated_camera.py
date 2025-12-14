import argparse
import glob
import os
import sys

import bpy
from mathutils import Matrix

# Reuse ensure_uv and bake logic patterns from projector.py
try:
    from projector import ensure_uv
except Exception:
    def ensure_uv(obj):
        me = obj.data
        if not hasattr(me, 'uv_layers'):
            return
        if len(me.uv_layers) == 0:
            bpy.context.view_layer.objects.active = obj
            for o in bpy.context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
            bpy.ops.object.mode_set(mode='OBJECT')


def _parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(description='Project a directory of images onto the meshes of an opened .blend using an animated camera')
    p.add_argument('--camera_name', required=True, help='Existing camera object name in the .blend')
    p.add_argument('--images', required=True, help='Directory or glob of images; order maps to sampled frames')
    p.add_argument('--start_frame', type=int, default=None, help='Start frame to sample; default scene.frame_start')
    p.add_argument('--end_frame', type=int, default=None, help='End frame to sample; default scene.frame_end')
    p.add_argument('--step', type=int, default=1, help='Frame step for sampling the camera trajectory')
    p.add_argument('--targets', nargs='*', default=None, help='Optional mesh names to project onto; default all meshes')
    p.add_argument('--uv_offset', nargs=2, type=float, default=(0.0, 0.0), help='UV offset for projection')
    p.add_argument('--bake_size', type=int, default=0, help='Square bake resolution; 0 uses first image size')
    p.add_argument('--pack', action='store_true', help='Pack baked images into blend')
    p.add_argument('--save_blend', default=None, help='Optional path to save the modified blend after baking')
    return p.parse_args(argv)


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
        raise FileNotFoundError(f'No images found from {spec}')
    return files


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


def _bake_once(obj, cam_obj, src_img_path, bake_img, uv_offset=(0.0, 0.0)):
    src_img = bpy.data.images.load(src_img_path, check_existing=True)
    ensure_uv(obj)
    uv_name = 'ProjectedUV'
    _write_projected_uv(obj, cam_obj, uv_name, uv_offset=uv_offset)

    mat = bpy.data.materials.new(name=f'ProjBake_{obj.name}')
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    uvmap_node = nodes.new('ShaderNodeUVMap')
    uvmap_node.uv_map = uv_name
    uvmap_node.location = (-600, 0)

    src_tex = nodes.new('ShaderNodeTexImage')
    src_tex.image = src_img
    src_tex.extension = 'CLIP'
    src_tex.interpolation = 'Linear'
    src_tex.location = (-300, 0)

    emit = nodes.new('ShaderNodeEmission')
    emit.location = (0, 0)

    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (300, 0)

    target_node = nodes.new('ShaderNodeTexImage')
    target_node.image = bake_img
    target_node.location = (-300, -200)
    nt.nodes.active = target_node

    links.new(uvmap_node.outputs['UV'], src_tex.inputs['Vector'])
    links.new(src_tex.outputs['Color'], emit.inputs['Color'])
    links.new(emit.outputs['Emission'], out.inputs['Surface'])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    scene = bpy.context.scene
    prev_engine = scene.render.engine
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles
    prev_bake_type = getattr(cycles, 'bake_type', None)
    cycles.bake_type = 'EMIT'

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type='EMIT', use_clear=False)

    scene.render.engine = prev_engine
    if prev_bake_type is not None:
        cycles.bake_type = prev_bake_type


def _assign_baked_material(obj, bake_img):
    mat = bpy.data.materials.new(name=f'Baked_{obj.name}')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.image = bake_img
    tex_node.location = (-300, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (300, 0)

    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def main():
    args = _parse_args()
    scene = bpy.context.scene
    # Use image set to determine bake resolution
    images = _collect_images(args.images)
    sample_img = bpy.data.images.load(images[0], check_existing=True)
    width, height = sample_img.size
    if args.bake_size > 0:
        width = height = int(args.bake_size)

    # Resolve camera
    if args.camera_name not in bpy.data.objects:
        raise RuntimeError(f"Camera '{args.camera_name}' not found in scene")
    cam_obj = bpy.data.objects[args.camera_name]

    # Frames to sample
    start = args.start_frame or scene.frame_start
    end = args.end_frame or scene.frame_end
    step = max(1, args.step)
    frames = list(range(start, end + 1, step))
    if len(images) < len(frames):
        print(f"Warning: fewer images ({len(images)}) than sampled frames ({len(frames)}); extra frames ignored")
    pairs = list(zip(frames, images))

    # Targets
    mesh_objs = [o for o in scene.objects if o.type == 'MESH'] if not args.targets else [scene.objects[n] for n in args.targets if n in scene.objects]
    if not mesh_objs:
        raise RuntimeError('No mesh objects to project onto')

    # Create bake images per object
    bake_images = {obj.name: bpy.data.images.new(name=f"bake_{obj.name}", width=width, height=height, alpha=True) for obj in mesh_objs}

    # Perform per-frame projection bake
    for idx, (f, img_path) in enumerate(pairs, start=1):
        print(f"Frame {f} -> {os.path.basename(img_path)}")
        scene.frame_set(int(f))
        for obj in mesh_objs:
            _bake_once(obj, cam_obj, img_path, bake_images[obj.name], uv_offset=tuple(args.uv_offset))

    # Assign baked textures
    for obj in mesh_objs:
        _assign_baked_material(obj, bake_images[obj.name])
        if args.pack:
            bake_images[obj.name].pack()
    if args.pack:
        bpy.ops.file.pack_all()

    # Optionally save blend
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.save_blend)
        print(f"Saved baked blend: {args.save_blend}")

    print('Done baking projections from animated camera.')


if __name__ == '__main__':
    main()
