# 推理框架

## 流水线测试

```bash
export PYTHONPATH=/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer
python -m mvadapter.test.pipeline \
  --device cuda \
  --uv-size 2048 \
  --frame-step 1 \
  --max-frames 100 \
  --blender-bin ./blender/blender-5.0.0-linux-x64/blender \
  --output-dir mvadapter/test/output5 \
  --debug
```

## 渲染测试

```bash
export PYTHONPATH=/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer
python -m mvadapter.test.render_pipeline \
  --device cuda \
  --uv-size 2048 \
  --frame-step 1 \
  --max-frames 100 \
  --blender-bin ./blender/blender-5.0.0-linux-x64/blender \
  --output-dir mvadapter/test/output5
```

## 导出相机视角轨迹

```bash
python -m mvadapter.test.export_camera \
  --device cuda \
  --uv-size 2048 \
  --frame-step 1 \
  --max-frames 100 \
  --blender-bin ./blender/blender-5.0.0-linux-x64/blender \
  --output-dir mvadapter/test/output5
```

## 图片投影测试

在项目根目录下

```bash
/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer/blender/blender-5.0.0-linux-x64/blender -b -P mvadapter/test/project_glb_with_cameras.py -- \
  --input_glb mvadapter/test/town.glb \
  --camera_json mvadapter/test/camera_path.json \
  --images mvadapter/test/frames \
  --output_glb mvadapter/test/projected_town.glb \
  --pack
```

## 图片渲染测试

在项目根目录下

```bash
/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer/blender/blender-5.0.0-linux-x64/blender mvadapter/test/town.blend -b -P mvadapter/test/render_cameras_from_glb.py -- \
  --glb mvadapter/test/projected_town.glb \
  --camera_json mvadapter/test/camera_path.json \
  --output_dir mvadapter/test/renders \
  --output_prefix view_rgb_ \
  --resolution 720 480 \
  --engine CYCLES \
  --cycles_samples 64
```

无显示环境下

```bash
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -s "-screen 0 1024x768x24" \
/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer/blender/blender-5.0.0-linux-x64/blender \
  mvadapter/test/town.blend -b -P mvadapter/test/render_cameras_from_glb.py -- \
  --glb mvadapter/test/projected_town.glb \
  --camera_json mvadapter/test/camera_path.json \
  --output_dir mvadapter/test/renders \
  --output_prefix view_rgb_ \
  --resolution 720 480 \
  --engine BLENDER_EEVEE \
  --cycles_samples 1 \
  --clear_scene \
  --sun --sun_strength 3.0 \
  --world_color 0.1 0.1 0.1
```