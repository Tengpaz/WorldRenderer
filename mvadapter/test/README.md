# 渲染器

## 指定blend文件、视频帧生成导出对应的数据图像

```bash
export PYTHONPATH=/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer # 指向项目根目录
python -m mvadapter.test.pipeline \
  --device cuda \
  --uv-size 2048 \
  --frame-step 1 \
  --max-frames 100 \
  --blender-bin ./blender/blender-5.0.0-linux-x64/blender \
  --blend-path mvadapter/test/town.blend \
  --video-path mvadapter/test/video.mp4 \
  --output-dir mvadapter/test/output \
  --debug
```

导出的depth和normal默认是按照给的视频帧的分辨率导出

- uv-size: 贴图分辨率
- frame-step: 帧步长
- max-frames: 最大生成帧数
- blender-bin: blender路径
- blend-path: blend文件路径
- video-path: 视频或视频帧所在目录路径
- output-dir: 输出目录

## 从blend导出depth和nornal数据

```bash
export PYTHONPATH=/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer # 指向项目根目录
python -m mvadapter.test.pipeline \
  --device cuda \
  --uv-size 2048 \
  --frame-step 1 \
  --max-frames 100 \
  --height 480 \
  --width 720 \
  --blender-bin ./blender/blender-5.0.0-linux-x64/blender \
  --blend-path mvadapter/test/town.blend \
  --output-dir mvadapter/test/output \
  --debug
```

## 视角到视角的流水线测试

```bash
export PYTHONPATH=/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer
python -m mvadapter.test.pipeline_view \
  --device cuda \
  --uv-size 4096 \
  --frame-step 1 \
  --max-frames 1 \
  --blender-bin ./blender/blender-5.0.0-linux-x64/blender \
  --output-dir mvadapter/test/output6 \
  --debug
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

## 文件说明

项目结构

```
mvadapter
- test/
  - pipeline.py # 导出depth normal rgb
  - pipeline_view.py # 单视角到单视角的反投渲染（废弃，已改为使用点云）
  - export_camera.py # 导出相机轨迹，用于调试
  - utils/ # 工具函数
- utils/
  - mesh_utils/
   - mesh.py
   - utils.py
   - render.py
   - camera.py
```