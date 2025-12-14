import bpy
from mathutils import Vector
import os
from os.path import join

def setup_depth_compositor(scene=None):
    """设置合成器来渲染深度通道，使用固定相机裁剪范围映射深度到0-1，避免帧间归一化闪烁"""
    if scene is None:
        scene = bpy.context.scene

    # 启用深度通道
    scene.view_layers["ViewLayer"].use_pass_z = True

    # 启用节点
    if not scene.use_nodes:
        scene.use_nodes = True

    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # 清空现有节点
    nodes.clear()

    # 渲染图层节点
    render_layers = nodes.new('CompositorNodeRLayers')
    render_layers.location = 0, 0 # 位置

    # 使用相机裁剪范围进行固定映射
    map_value = nodes.new('CompositorNodeMapValue')
    map_value.location = 200, 0 # 位置

    # 初始裁剪范围(优先使用场景基础范围（对象边界框）)
    clip_start = 0.1
    clip_end = 100.0
    cam = scene.camera
    if cam and hasattr(cam, 'data'):
        try:
            # 相机裁剪范围作为安全回退
            cam_clip_start = float(cam.data.clip_start)
            cam_clip_end = float(cam.data.clip_end)
            if cam_clip_end > cam_clip_start:
                clip_start = cam_clip_start
                clip_end = cam_clip_end
        except Exception:
            clip_start = 0.1
            clip_end = 100.0

    # 避免使用场景中对象的边界框来计算更紧凑、稳定的深度范围
    try:
        if cam:
            cam_loc = cam.matrix_world.to_translation() # 相机位置
        else:
            cam_loc = Vector((0.0, 0.0, 0.0)) # 默认位置
        min_d = float('inf')
        max_d = 0.0
        for obj in scene.objects:
            if obj.type != 'MESH':
                continue
            # 边缘框是局部空间中的8个角点
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                d = (world_corner - cam_loc).length
                if d > 0.0:
                    min_d = min(min_d, d)
                    max_d = max(max_d, d)

        if min_d != float('inf') and max_d > 0.0:
            # expand range slightly to avoid clipping exactly on surfaces
            span_obj = max_d - min_d
            if span_obj < 1e-4:
                # fallback to camera clip if objects are almost at same distance
                span_obj = max(1.0, span_obj)
            pad = span_obj * 0.05
            clip_start = max(0.0001, min_d - pad)
            clip_end = max(clip_start + 1e-4, max_d + pad)
    except Exception:
        pass

    span = max(clip_end - clip_start, 1e-6)
    size = 1.0 / span
    offset = -clip_start * size

    # Map Value uses arrays for offset/size; set values using the node's array length
    try:
        count = len(map_value.offset)
    except Exception:
        # fallback to 1 channel if the property isn't a sequence
        count = 1

    # Try assigning the whole arrays first (works when Blender accepts list assignment)
    try:
        map_value.offset = [offset] * count
        map_value.size = [size] * count
        map_value.min = [0.0] * count
        map_value.max = [1.0] * count
    except Exception:
        # Fallback: assign per-index up to the available count
        for i in range(count):
            try:
                map_value.offset[i] = offset
            except Exception:
                pass
            try:
                map_value.size[i] = size
            except Exception:
                pass
            try:
                map_value.min[i] = 0.0
            except Exception:
                pass
            try:
                map_value.max[i] = 1.0
            except Exception:
                pass

    map_value.use_min = True
    map_value.use_max = True

    composite = nodes.new('CompositorNodeComposite')
    composite.location = 400, 0

    links.new(render_layers.outputs['Depth'], map_value.inputs[0])
    links.new(map_value.outputs[0], composite.inputs['Image'])

    print("深度合成器设置完成（使用固定相机裁剪范围映射）")

def export():
    scene = bpy.context.scene
    root = os.path.dirname(bpy.data.filepath)

    # 设置渲染参数
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW' # 黑白图
    scene.render.image_settings.color_depth = '16' # 16位深度
    scene.render.use_file_extension = True # 使用文件扩展名
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 1 # 采样数降低以加速渲染

    # 设置深度合成器
    setup_depth_compositor(scene)

    # 创建深度图目录
    depth_dir = join(root, 'depth')
    os.makedirs(depth_dir, exist_ok=True) # 确保目录存在

    # 渲染每一帧
    start_frame = scene.frame_start
    end_frame = scene.frame_end

    for frame in range (start_frame, end_frame + 1):
        scene.frame_set(frame) # 设置当前帧
        output_path = join(depth_dir, f"{frame:04d}.png")
        scene.render.filepath = output_path
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            print(f"渲染失败帧 {frame}: {e}")