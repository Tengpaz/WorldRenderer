
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
        'matrix_world': [[float(mw[i][j]) for j in range(4)] for i in range(4)]
    })
with open(r"/apdcephfs_cq5/share_300600172/suanhuang/users/wangyuzhen/WorldRenderer/test/camera_path.json", 'w') as fp:
    json.dump(data, fp)
