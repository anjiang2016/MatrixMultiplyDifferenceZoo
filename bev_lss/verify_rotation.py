# verify_rotation.py
import json
import numpy as np
import os

DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"
CALIB_FILE = "v1.0-mini/calibrated_sensor.json"

# 目标前摄像头标定 token (根据你提供的数据)
TARGET_CALIB_TOKEN = "1d31c729b073425e8e0202c5c6e66ee1"

def load_json(rel_path):
    with open(os.path.join(DATA_ROOT, rel_path), 'r') as f:
        return json.load(f)

def quat_to_rot(q):
    """四元数 (w,x,y,z) → 旋转矩阵 (3x3)"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,       2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ])

def main():
    # 1. 加载标定数据
    calib_data = load_json(CALIB_FILE)
    target_entry = None
    for entry in calib_data:
        if entry['token'] == TARGET_CALIB_TOKEN:
            target_entry = entry
            break
    if target_entry is None:
        print(f"未找到标定 token: {TARGET_CALIB_TOKEN}")
        return

    # 2. 提取数据
    rot_quat = target_entry['rotation']   # [w, x, y, z]
    trans = target_entry['translation']   # [x, y, z]
    intrin = target_entry.get('camera_intrinsic', None)

    print("===== 前摄像头标定信息 =====")
    print(f"四元数 (w, x, y, z): {rot_quat}")
    print(f"平移向量 (x, y, z): {trans}")
    if intrin:
        print(f"内参矩阵:\n{np.array(intrin)}")

    # 3. 计算旋转矩阵
    R = quat_to_rot(rot_quat)
    print("\n===== 旋转矩阵 R =====")
    print(R)

    # 4. 验证相机轴 → 车辆轴
    # 相机坐标系: x右, y下, z前
    cam_x = np.array([1.0, 0.0, 0.0])   # 相机右
    cam_y = np.array([0.0, 1.0, 0.0])   # 相机下
    cam_z = np.array([0.0, 0.0, 1.0])   # 相机前

    ego_x = R @ cam_x
    ego_y = R @ cam_y
    ego_z = R @ cam_z

    print("\n===== 相机轴 → 车辆轴 =====")
    print(f"相机右 (1,0,0) → 车辆: {ego_x}")
    print(f"相机下 (0,1,0) → 车辆: {ego_y}")
    print(f"相机前 (0,0,1) → 车辆: {ego_z}")

    # 5. 归一化方向（应该接近 (1,0,0), (0,1,0), (0,0,1) 或它们的负组合）
    print("\n===== 预期车辆轴（nuScenes: x前, y左, z上）=====")
    print("车辆前方应该是 (1,0,0)")
    print("车辆左侧应该是 (0,1,0)")
    print("车辆上方应该是 (0,0,1)")

    # 检查模长
    print(f"\n模长验证: ||R*(0,0,1)|| = {np.linalg.norm(ego_z):.6f}")

    # 6. 额外: 验证 R 是正交矩阵
    I_check = R @ R.T
    print("\n正交性验证 (R*R^T 应接近单位阵):")
    print(I_check)

if __name__ == "__main__":
    main()
