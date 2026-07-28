# check_annotation_pose.py
import json
import numpy as np
import os

DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"

def load_json(rel_path):
    with open(os.path.join(DATA_ROOT, rel_path), 'r') as f:
        return json.load(f)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

# 加载一个样本的标注和 ego_pose
samples = load_json('v1.0-mini/sample.json')
sample_token = samples[0]['token']
print(f"样本 token: {sample_token}")

# 获取 ego_pose
sample_data_list = load_json('v1.0-mini/sample_data.json')
ego_pose_token = None
for sd in sample_data_list:
    if sd['sample_token'] == sample_token and sd['is_key_frame']:
        ego_pose_token = sd['ego_pose_token']
        break
if ego_pose_token is None:
    raise ValueError("未找到 ego_pose_token")

ego_poses = load_json('v1.0-mini/ego_pose.json')
ego_pose_map = {ep['token']: ep for ep in ego_poses}
ego_pose = ego_pose_map[ego_pose_token]
R_ego2global = quaternion_to_rotation_matrix(ego_pose['rotation'])
t_ego2global = np.array(ego_pose['translation'])
R_global2ego = R_ego2global.T
t_global2ego = -R_global2ego @ t_ego2global

print("R_ego2global (自车→全局):\n", R_ego2global)
print("t_ego2global (自车在全局的位置):", t_ego2global)
print("R_global2ego (全局→自车):\n", R_global2ego)
print("t_global2ego (全局原点在自车系的位置):", t_global2ego)

# 加载车辆标注
sample_anns = load_json('v1.0-mini/sample_annotation.json')
instances = load_json('v1.0-mini/instance.json')
categories = load_json('v1.0-mini/category.json')
instance_to_category = {inst['token']: inst['category_token'] for inst in instances}
cat_name_map = {cat['token']: cat['name'] for cat in categories}

def get_category(ann):
    inst_token = ann['instance_token']
    if inst_token in instance_to_category:
        cat_token = instance_to_category[inst_token]
        return cat_name_map.get(cat_token, 'unknown')
    return 'unknown'

# 取前两个车辆标注
vehicle_anns = []
for ann in sample_anns:
    if ann['sample_token'] == sample_token:
        cat = get_category(ann)
        if cat.startswith('vehicle.'):
            vehicle_anns.append((ann, cat))
            if len(vehicle_anns) >= 2:
                break

if not vehicle_anns:
    print("未找到车辆标注")
    exit()

for idx, (ann, cat) in enumerate(vehicle_anns):
    print(f"\n=== 车辆 {idx+1} ({cat}) ===")
    pos_global = np.array(ann['translation'])
    quat = ann['rotation']  # [w, x, y, z]
    # 全局偏航角
    yaw_global = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                            1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    dir_global = np.array([np.cos(yaw_global), np.sin(yaw_global), 0])
    print(f"全局位置: {pos_global}")
    print(f"全局偏航角: {yaw_global} rad ({np.degrees(yaw_global):.1f}°)")
    print(f"全局方向向量: {dir_global}")

    # 转换到自车
    R_global2ego = ego_pose_mat[:3,:3].T
    pos_ego = R_global2ego @ pos_global + t_global2ego
    dir_ego = np.array([np.cos(yaw_global), np.sin(yaw_global),0])
    dir_ego = R_global2ego @ dir_global
    yaw_ego = np.arctan2(dir_ego[1], dir_ego[0])
    print(f"自车位置: {pos_ego}")
    print(f"自车方向向量: {dir_ego}")
    print(f"自车偏航角: {yaw_ego} rad ({np.degrees(yaw_ego):.1f}°)")

    # 简单检查：如果车辆在自车前方，pos_ego.x 应为正，方向也应大致朝前。
