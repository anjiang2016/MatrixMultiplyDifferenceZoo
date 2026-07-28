import json
import numpy as np
import os

DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"

def load_json(rel_path):
    with open(os.path.join(DATA_ROOT, rel_path), 'r') as f:
        return json.load(f)

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,       2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ])

def get_cam_front_image_info():
    sensors = load_json('v1.0-mini/sensor.json')
    sensor_token_to_channel = {s['token']: s['channel'] for s in sensors}

    calib_sensors = load_json('v1.0-mini/calibrated_sensor.json')
    calib_map = {}
    for cs in calib_sensors:
        sensor_token = cs['sensor_token']
        channel = sensor_token_to_channel.get(sensor_token)
        if channel != 'CAM_FRONT':
            continue
        intrin = np.array(cs.get('camera_intrinsic', np.eye(3)))
        R_ego2cam = quat_to_rot(cs['rotation'])
        t_ego2cam = np.array(cs['translation'])
        # 相机→自车：旋转取逆，平移直接用（因为平移是传感器原点在自车系的位置）
        R_cam2ego = R_ego2cam.T
        t_cam2ego = t_ego2cam  # 直接使用
        calib_map[cs['token']] = {
            'intrin': intrin,
            'R_cam2ego': R_cam2ego,
            'R_ego2cam': R_ego2cam,
            't_cam2ego': t_cam2ego
        }

    if not calib_map:
        raise ValueError("未找到 CAM_FRONT 标定")

    sample_data_list = load_json('v1.0-mini/sample_data.json')
    for sd in sample_data_list:
        if not sd.get('is_key_frame', False):
            continue
        calib_token = sd['calibrated_sensor_token']
        if calib_token not in calib_map:
            continue
        img_path = os.path.join(DATA_ROOT, sd['filename'])
        sample_token = sd['sample_token']
        info = calib_map[calib_token]
        return sample_token, img_path, info['intrin'], info['R_cam2ego'], info['R_ego2cam'],info['t_cam2ego']

    raise ValueError("未找到 CAM_FRONT 关键帧")

def main():
    sample_token, img_path, intrin, R_cam2ego, R_ego2cam,t_cam2ego = get_cam_front_image_info()
    print(f"样本: {sample_token}")
    print(f"图像路径: {img_path}")
    print("内参:\n", intrin)
    print("R (相机→自车):\n", R_cam2ego)
    print("t (相机→自车):", t_cam2ego)

    points = {
        "中心": (800, 450),
        "左上": (0, 0),
        "右上": (1599, 0),
        "左下": (0, 899),
        "右下": (1599, 899),
    }

    print("\n===== 逐步变换 =====")
    for name, (u, v) in points.items():
        # 1. 归一化平面坐标
        x_norm = (u - intrin[0,2]) / intrin[0,0]
        y_norm = (v - intrin[1,2]) / intrin[1,1]
        # 2. 相机坐标系下的射线方向（z=1）
        ray_cam = np.array([x_norm, y_norm, 1.0])

        # 3. 旋转到自车坐标系（仅旋转，不加平移）
        ray_ego_rot = R_ego2cam @ ray_cam

        # 4. 加平移并求地面交点（z=0）
        # P_ego = λ * ray_ego_rot + t_cam2ego, 令 z=0
        denom = ray_ego_rot[2]
        if abs(denom) < 1e-8:
            ground_point = None
        else:
            lam = -t_cam2ego[2] / denom
            ground_point = lam * ray_ego_rot + t_cam2ego

        print(f"\n{name} ({u},{v}):")
        print(f"  归一化坐标: ({x_norm:.6f}, {y_norm:.6f})")
        print(f"  相机射线 (相机坐标系): ({ray_cam[0]:.6f}, {ray_cam[1]:.6f}, {ray_cam[2]:.6f})")
        print(f"  旋转后方向 (自车坐标系): ({ray_ego_rot[0]:.6f}, {ray_ego_rot[1]:.6f}, {ray_ego_rot[2]:.6f})")
        if ground_point is not None:
            print(f"  地面交点 (自车坐标系): ({ground_point[0]:.6f}, {ground_point[1]:.6f}, {ground_point[2]:.6f})")
        else:
            print("  地面交点: 无 (射线平行于地面)")

if __name__ == "__main__":
    main()
