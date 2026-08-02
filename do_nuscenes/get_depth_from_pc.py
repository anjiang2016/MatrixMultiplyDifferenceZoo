import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyquaternion import Quaternion

CAM_CHANNEL = 'CAM_BACK'
# ========== 1. 加载 JSON 文件并建立索引 ==========
def load_json_file(json_dir, filename):
    with open(os.path.join(json_dir, filename), 'r') as f:
        return json.load(f)

def build_index(data_list, key='token'):
    """根据 token 建立字典索引"""
    return {item[key]: item for item in data_list}

# 加载所有 JSON 文件
json_dir = "/Users/zhaomingming/data_sets/v1.0-mini/v1.0-mini/"  # 替换为你的 JSON 文件夹
dataroot = "/Users/zhaomingming/data_sets/v1.0-mini/"          # 替换为数据根目录

sample_data_list = load_json_file(json_dir, 'sample_data.json')
calibrated_sensor_list = load_json_file(json_dir, 'calibrated_sensor.json')
ego_pose_list = load_json_file(json_dir, 'ego_pose.json')
sample_list = load_json_file(json_dir, 'sample.json')
sensor_list = load_json_file(json_dir,'sensor.json')
# ========== 加载 scene.json ==========
scene_list = load_json_file(json_dir, 'scene.json')

# 建立索引
sample_data_idx = build_index(sample_data_list)
calibrated_sensor_idx = build_index(calibrated_sensor_list)
ego_pose_idx = build_index(ego_pose_list)
sensor_idx = build_index(sensor_list)

# ========== 直接通过 sample_token 在 sample_data 中查找 ==========
# 1. 从 scene.json 拿到第一个场景的第一个 sample_token
first_scene = scene_list[0]
target_sample_token = first_scene['first_sample_token']

# 2. 在 sample_data 中按 sample_token 过滤，分别提取 CAM_FRONT 和 LIDAR_TOP
cam_data = None
lidar_data = None

for sd in sample_data_list:
    # 只处理属于该 sample 的数据
    if sd.get('sample_token') != target_sample_token:
        continue
    
    # 通过标定 -> 传感器链获取 channel
    calib = calibrated_sensor_idx.get(sd['calibrated_sensor_token'])
    if calib is None:
        continue
    sensor = sensor_idx.get(calib.get('sensor_token'))
    if sensor is None:
        continue
    channel = sensor.get('channel')
    
    if channel == CAM_CHANNEL:
        cam_data = sd
    elif channel == 'LIDAR_TOP':
        lidar_data = sd

# 3. 检查是否都找到了
if cam_data is None or lidar_data is None:
    print(f"在 sample_token: {target_sample_token} 下未同时找到 CAM_FRONT 和 LIDAR_TOP")
    # 可打印该 sample 下实际有哪些通道，方便调试
    available = []
    for sd in sample_data_list:
        if sd.get('sample_token') == target_sample_token:
            calib = calibrated_sensor_idx.get(sd['calibrated_sensor_token'])
            if calib:
                sensor = sensor_idx.get(calib.get('sensor_token'))
                if sensor:
                    available.append(sensor.get('channel'))
    print(f"实际可用的通道: {available}")
    exit()

print(f"成功匹配关键帧!")
print(f"Sample token: {target_sample_token}")
print(f"Image file: {cam_data['filename']}")
print(f"Lidar file: {lidar_data['filename']}")

# ========== 3. 读取标定和位姿信息 ==========
def get_calibration(calib_token):
    calib = calibrated_sensor_idx.get(calib_token)
    if calib is None:
        return None, None, None
    rot = calib['rotation']  # [w, x, y, z]
    trans = calib['translation']  # [x, y, z]
    intrinsic = calib.get('camera_intrinsic')  # 仅相机有
    return rot, trans, intrinsic

def get_ego_pose(pose_token):
    pose = ego_pose_idx.get(pose_token)
    if pose is None:
        return None, None
    return pose['rotation'], pose['translation']

# 相机标定
cam_rot, cam_trans, cam_intrinsic = get_calibration(cam_data['calibrated_sensor_token'])
# 雷达标定
lidar_rot, lidar_trans, _ = get_calibration(lidar_data['calibrated_sensor_token'])
# 相机 ego_pose
cam_ego_rot, cam_ego_trans = get_ego_pose(cam_data['ego_pose_token'])
# 雷达 ego_pose
lidar_ego_rot, lidar_ego_trans = get_ego_pose(lidar_data['ego_pose_token'])

# ========== 4. 加载点云和图像 ==========
def load_pointcloud(pcd_path):
    # 点云为二进制 float32，每点 x,y,z,intensity
    raw = np.fromfile(pcd_path, dtype=np.float32)
    points = raw.reshape(-1, 4)[:, :3].T  # (3, N)
    return points

img_path = os.path.join(dataroot, cam_data['filename'])
pcd_path = os.path.join(dataroot, lidar_data['filename'])

img = Image.open(img_path)
img_width, img_height = img.size
points = load_pointcloud(pcd_path)
print(f"Image size: {img_width} x {img_height}, Lidar points: {points.shape[1]}")

# ========== 5. 投影函数（四元数转旋转矩阵） ==========
def quat_to_rot(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

def project_lidar_to_image(points_3xN,
                           cam_intrinsic,
                           cam_rot, cam_trans,
                           lidar_rot, lidar_trans,
                           ego_lidar_rot, ego_lidar_trans,
                           ego_cam_rot, ego_cam_trans,
                           img_w, img_h):
    # 1. 雷达 -> 自车 (雷达时刻)
    R_lidar_ego = quat_to_rot(lidar_rot)
    pts = R_lidar_ego @ points_3xN + np.array(lidar_trans).reshape(3,1)

    # 2. 自车 -> 全局 (雷达时刻)
    R_ego_global = quat_to_rot(ego_lidar_rot)
    pts = R_ego_global @ pts + np.array(ego_lidar_trans).reshape(3,1)

    # 3. 全局 -> 自车 (相机时刻)
    R_global_ego = quat_to_rot(ego_cam_rot).T  # 逆旋转
    pts = R_global_ego @ (pts - np.array(ego_cam_trans).reshape(3,1))

    # 4. 自车 -> 相机 (相机坐标系)
    R_ego_cam = quat_to_rot(cam_rot).T
    pts = R_ego_cam @ (pts - np.array(cam_trans).reshape(3,1))

    depths = pts[2, :]  # 深度

    # 5. 投影到图像
    img_pts = cam_intrinsic @ pts  # (3, N)
    u = img_pts[0, :] / img_pts[2, :]
    v = img_pts[1, :] / img_pts[2, :]

    # 6. 过滤
    mask = (depths > 0) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    mask &= np.isfinite(u) & np.isfinite(v) & np.isfinite(depths)

    return np.vstack([u[mask], v[mask]]), depths[mask]

# 执行投影
pixel_coords, depths = project_lidar_to_image(
    points,
    np.array(cam_intrinsic),
    cam_rot, cam_trans,
    lidar_rot, lidar_trans,
    lidar_ego_rot, lidar_ego_trans,
    cam_ego_rot, cam_ego_trans,
    img_width, img_height
)

print(f"投影成功点数量: {pixel_coords.shape[1]}")

# ========== 6. 可视化 ==========
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(img)
sc = ax.scatter(pixel_coords[0, :], pixel_coords[1, :],
                c=depths, s=1, cmap='jet', vmin=0, vmax=50)
plt.colorbar(sc, label='Depth (m)')
ax.set_title("Lidar projection to "+CAM_CHANNEL+" image (color = depth)")
plt.show()
