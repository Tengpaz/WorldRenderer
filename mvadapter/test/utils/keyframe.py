import json
import numpy as np

def calculate_position_diff(P1, P2):
    """计算相机位置差异"""
    return np.linalg.norm(np.array(P1) - np.array(P2))

def calculate_rotation_diff(R1, R2):
    """计算旋转矩阵差异"""
    R1 = np.array(R1)
    R2 = np.array(R2)
    R1 = R1[:3, :3]
    R2 = R2[:3, :3]
    U, _, Vt = np.linalg.svd(R1)
    R1 = np.dot(U, Vt)
    U, _, Vt = np.linalg.svd(R2)
    R2 = np.dot(U, Vt)
    
    # 计算旋转矩阵之间的差异
    trace = np.trace(np.dot(np.array(R1).T, np.array(R2)))
    return np.arccos((trace - 1) / 2)

def load_camera_data(file_path):
    """从json文件加载相机数据"""
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    return data

def is_keyframe(prev_frame, curr_frame, t_position, t_rotation):
    """根据位置和旋转的差异判断是否为关键帧"""
    P1, R1 = prev_frame['matrix_world'][:3][-1], prev_frame['matrix_world'][:3][:3]
    P2, R2 = curr_frame['matrix_world'][:3][-1], curr_frame['matrix_world'][:3][:3]
    
    dist_position = calculate_position_diff(P1, P2)
    dist_rotation = calculate_rotation_diff(R1, R2)
    print(f"位置差异: {dist_position}, 旋转差异: {dist_rotation}")

    if dist_position < 0.1:
        return dist_rotation > t_rotation
    else:
        return dist_position > t_position or dist_rotation > t_rotation

def extract_keyframes(camera_data, t_position=0.5, t_rotation=0.1):
    """从相机数据中提取关键帧"""
    keyframes = [0]  # 第一个帧默认是关键帧
    
    for i in range(1, len(camera_data)):
        if is_keyframe(camera_data[keyframes[-1]], camera_data[i], t_position, t_rotation):
            print(f"关键帧: {i}, 参照帧: {keyframes[-1]}")
            keyframes.append(i)
    
    return keyframes

def save_keyframes(output_file, keyframes):
    """保存提取的关键帧列表"""
    keyframe_indices = [frame['frame'] for frame in keyframes]
    with open(output_file, 'w') as fp:
        json.dump(keyframe_indices, fp)
    print(f"关键帧已保存到 {output_file}")

def get_keyframes(json_path, num_views, t_position=0.5, t_rotation=0.1):
    """主函数：加载相机数据，提取关键帧并保存"""
    camera_data = load_camera_data(json_path)
    camera_data = camera_data[:num_views]
    keyframes = extract_keyframes(camera_data, t_position, t_rotation)
    return keyframes

# camera_data = load_camera_data(r"mvadapter/test/output/camera.json")
# keyframes = extract_keyframes(camera_data, t_position=0.5, t_rotation=0.1)
# save_keyframes(r"mvadapter/test/output/keyframes.json", keyframes)