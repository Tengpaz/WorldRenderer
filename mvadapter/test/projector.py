import bpy
import os


def ensure_uv(obj):
    """Ensure the object has a UV map; if not, create one using Smart UV Project."""
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


def project_image_to_object_solid(camera_name, object_name, image_path, bake_save_path=None, uv_offset=(0.0, 0.0)):
    """Project `image_path` onto `object_name` from `camera_name` and bake (solidify) that projection

    This creates a projection shader that samples the input image using Camera texture
    coordinates, bakes the projected result into a new image (using the object's UVs),
    assigns the baked image to the object's material, and (optionally) saves the baked image
    to `bake_save_path`.
    """
    scene = bpy.context.scene

    # Validate
    if camera_name not in bpy.data.objects:
        raise RuntimeError(f"Camera '{camera_name}' not found")
    cam = bpy.data.objects[camera_name]
    if object_name not in bpy.data.objects:
        raise RuntimeError(f"Object '{object_name}' not found")
    obj = bpy.data.objects[object_name]
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image file not found: {image_path}")

    # Load source image
    src_img = bpy.data.images.load(image_path)
    width, height = src_img.size[0], src_img.size[1]

    # Ensure object has UVs
    ensure_uv(obj)

    # Create bake target image
    bake_img_name = f"bake_{object_name}_{src_img.name}"
    bake_img = bpy.data.images.new(bake_img_name, width=width, height=height, alpha=True)

    # Create a new material for projection-baking that samples the image via a UV map
    mat = bpy.data.materials.new(name=f"ProjectionBakeMat_{object_name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    # We'll create (or overwrite) a UV map named 'ProjectedUV' that is generated from the camera
    projected_uv_name = 'ProjectedUV'
    # Ensure the mesh has a UV layer we can target
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name=projected_uv_name)
    else:
        # create or ensure the named UV layer exists
        if projected_uv_name not in obj.data.uv_layers:
            obj.data.uv_layers.new(name=projected_uv_name)
    # Instead of relying on the UVProject modifier (which can be imprecise
    # or get applied differently across Blender versions), compute camera-projected
    # UVs directly and write them into the 'ProjectedUV' layer for exact alignment.
    def camera_project_to_uv(target_obj, camera_obj, uv_name=projected_uv_name, uv_offset_local=(0.0, 0.0)):
        from bpy_extras.object_utils import world_to_camera_view
        mesh = target_obj.data
        # Ensure the UV layer exists
        if uv_name not in mesh.uv_layers:
            mesh.uv_layers.new(name=uv_name)
        uv_layer = mesh.uv_layers[uv_name]

        scene_ref = bpy.context.scene

        # Cache vertex world coords
        vert_world = [target_obj.matrix_world @ v.co for v in mesh.vertices]

        # For each loop (face corner), assign UV based on the projected vertex
        for loop in mesh.loops:
            vidx = loop.vertex_index
            world_co = vert_world[vidx]
            co_ndc = world_to_camera_view(scene_ref, camera_obj, world_co)
            # co_ndc.x, co_ndc.y are normalized [0,1] with origin at bottom-left
            u = float(co_ndc.x)
            v = float(co_ndc.y)
            # Convert to UV space (v = 1 - y) so images align with Blender's image origin
            v = 1.0 - v
            # apply optional offset (in UV units)
            u += float(uv_offset_local[0])
            v += float(uv_offset_local[1])
            # Clamp to 0..1
            if u < 0.0:
                u = 0.0
            if u > 1.0:
                u = 1.0
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            uv_layer.data[loop.index].uv = (u, v)

        # Make the new UV layer active
        mesh.uv_layers.active = uv_layer

    try:
        camera_obj = bpy.data.objects[camera_name]
        camera_project_to_uv(obj, camera_obj, projected_uv_name, uv_offset_local=uv_offset)
    except Exception as e:
        print(f"Warning: failed to write camera-projected UVs: {e}")

    # Build a simple emission material that samples the image via the UV map
    uvmap_node = nodes.new('ShaderNodeUVMap')
    uvmap_node.uv_map = obj.data.uv_layers.active.name
    uvmap_node.location = (-600, 0)

    src_tex = nodes.new('ShaderNodeTexImage')
    src_tex.image = src_img
    src_tex.interpolation = 'Linear'
    src_tex.extension = 'CLIP'
    src_tex.location = (-300, 0)

    emit = nodes.new('ShaderNodeEmission')
    emit.location = (0, 0)
    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (300, 0)

    # Target image node (this is where the bake will be written)
    target_node = nodes.new('ShaderNodeTexImage')
    target_node.image = bake_img
    target_node.location = (-300, -200)
    nt.nodes.active = target_node

    links.new(uvmap_node.outputs['UV'], src_tex.inputs['Vector'])
    links.new(src_tex.outputs['Color'], emit.inputs['Color'])
    links.new(emit.outputs['Emission'], out.inputs['Surface'])

    # Assign material to object (replace all existing materials with this one)
    if obj.type != 'MESH':
        raise RuntimeError('Target object must be a mesh')
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # Ensure Cycles engine for baking
    prev_engine = scene.render.engine
    scene.render.engine = 'CYCLES'

    # Bake settings
    cycles = scene.cycles
    prev_bake_type = getattr(cycles, 'bake_type', None)
    cycles.bake_type = 'EMIT'

    # Select object and make it active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Perform bake (EMIT) — result writes into the active image node (target_node)
    bpy.ops.object.bake(type='EMIT')

    # Restore render engine and bake type
    scene.render.engine = prev_engine
    if prev_bake_type is not None:
        cycles.bake_type = prev_bake_type

    # Save baked image if requested
    if bake_save_path:
        bake_img.filepath_raw = bake_save_path
        bake_img.file_format = 'PNG'
        bake_img.save()

    # Update material to use baked texture as Base Color (safer for final renders)
    # Build a simple Principled BSDF network that uses the baked image
    nodes.clear()
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.image = bake_img
    tex_node.location = (-400, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (-100, 0)
    mout = nodes.new('ShaderNodeOutputMaterial')
    mout.location = (200, 0)
    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], mout.inputs['Surface'])

    print(f"Projection baked into image: {bake_img.name}")
    return bake_img


if __name__ == '__main__':
    # convenience CLI when run directly inside Blender (for debugging)
    BLEND_FILE = os.environ.get('BLEND_PATH', '')
    CAMERA_OBJ_NAME = os.environ.get('CAMERA_NAME', '')
    TARGET_OBJ_NAME = os.environ.get('TARGET_OBJ_NAME', 'CenterCube')
    IMAGE_FILE = os.environ.get('TEST_IMAGE', '')
    OUT_DIR = os.environ.get('RENDERER_OUTPUT', '.')
    outpath = None
    if OUT_DIR and IMAGE_FILE:
        fname = os.path.splitext(os.path.basename(IMAGE_FILE))[0]
        outpath = os.path.join(OUT_DIR, f"baked_{TARGET_OBJ_NAME}_{fname}.png")
    if CAMERA_OBJ_NAME and TARGET_OBJ_NAME and IMAGE_FILE:
        project_image_to_object_solid(CAMERA_OBJ_NAME, TARGET_OBJ_NAME, IMAGE_FILE, bake_save_path=outpath)