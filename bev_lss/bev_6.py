import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
from scipy.ndimage import distance_transform_edt

# ========== 配置 ==========
DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
BEV_RANGE = (-20, 20)      # 米
BEV_RESOLUTION = 0.1       # 米/像素
BEV_SIZE = int((BEV_RANGE[1] - BEV_RANGE[0]) / BEV_RESOLUTION)

def load_json(rel_path):
    with open(os.path.join(DATA_ROOT, rel_path), 'r') as f:
        return json.load(f)

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def get_camera_calib(channel):
    """根据channel获取该相机的标定（内参、R_cam2ego、t_cam2ego）"""
    sensors = load_json('v1.0-mini/sensor.json')
    sensor_token_to_channel = {s['token']: s['channel'] for s in sensors}
    calib_sensors = load_json('v1.0-mini/calibrated_sensor.json')
    for cs in calib_sensors:
        sensor_token = cs['sensor_token']
        if sensor_token not in sensor_token_to_channel:
            continue
        if sensor_token_to_channel[sensor_token] != channel:
            continue
        intrin = np.array(cs['camera_intrinsic'])
        R_ego2cam = quat_to_rot(cs['rotation'])
        t_ego2cam = np.array(cs['translation'])
        # 相机→自车
        R_cam2ego = R_ego2cam.T
        t_cam2ego = t_ego2cam   # 直接使用（已验证）
        return intrin, R_cam2ego, t_cam2ego
    raise ValueError(f"未找到 {channel} 标定")

def get_sample_image(channel, sample_token=None):
    """获取某个样本中指定相机的图像路径"""
    sample_data = load_json('v1.0-mini/sample_data.json')
    if sample_token is None:
        # 取第一个有该相机关键帧的样本
        for sd in sample_data:
            if sd.get('is_key_frame') and sd.get('filename', '').startswith(f'samples/{channel}'):
                sample_token = sd['sample_token']
                break
    if sample_token is None:
        raise ValueError("未找到有效样本")
    for sd in sample_data:
        if (sd['sample_token'] == sample_token and sd.get('is_key_frame') and
            sd.get('filename', '').startswith(f'samples/{channel}')):
            return os.path.join(DATA_ROOT, sd['filename'])
    raise ValueError(f"未找到 {channel} 在样本 {sample_token} 中的图像")

def project_cam_to_bev(image, intrin, R_cam2ego, t_cam2ego,
                       bev_range, bev_res, step=2):
    """单相机投影到BEV（返回累加器和计数器）"""
    H, W, _ = image.shape
    fx, fy = intrin[0,0], intrin[1,1]
    cx, cy = intrin[0,2], intrin[1,2]
    bev_size = int((bev_range[1] - bev_range[0]) / bev_res)
    bev_acc = np.zeros((bev_size, bev_size, 3), dtype=np.float32)
    bev_cnt = np.zeros((bev_size, bev_size), dtype=np.uint16)

    for v in range(0, H, step):
        for u in range(0, W, step):
            x_c = (u - cx) / fx
            y_c = (v - cy) / fy
            ray_cam = np.array([x_c, y_c, 1.0])
            d_ego = R_cam2ego @ ray_cam
            if d_ego[2] >= 0:
                continue
            lam = -t_cam2ego[2] / d_ego[2]
            P_ego = lam * d_ego + t_cam2ego
            x_ego, y_ego = P_ego[0], P_ego[1]
            if x_ego < bev_range[0] or x_ego >= bev_range[1] or y_ego < bev_range[0] or y_ego >= bev_range[1]:
                continue
            gx = int((x_ego - bev_range[0]) / bev_res)
            gy = int((y_ego - bev_range[0]) / bev_res)
            gx = bev_size - 1 - gx  # 翻转x使前方在上
            if 0 <= gx < bev_size and 0 <= gy < bev_size:
                bev_acc[gx, gy] += image[v, u]
                bev_cnt[gx, gy] += 1
    return bev_acc, bev_cnt

def fill_bev_holes(bev, max_distance=5):
    mask = (bev.sum(axis=-1) > 0).astype(np.uint8)
    if np.sum(mask) == 0:
        return bev
    dist, idx = distance_transform_edt(mask, return_indices=True)
    filled = np.zeros_like(bev)
    for c in range(3):
        filled[:, :, c] = bev[:, :, c][idx[0], idx[1]]
    filled[dist > max_distance] = 0
    return filled

def generate_bev_panorama(sample_token=None, step=2, fill_holes=True):
    """生成多相机融合的BEV全景图"""
    # 初始化BEV
    bev_acc = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.float32)
    bev_cnt = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint16)

    # 对每个相机处理
    for cam in CAMERAS:
        print(f"处理 {cam}...")
        try:
            intrin, R_cam2ego, t_cam2ego = get_camera_calib(cam)
            img_path = get_sample_image(cam, sample_token)
            image = mpimg.imread(img_path)
            acc, cnt = project_cam_to_bev(image, intrin, R_cam2ego, t_cam2ego,
                                          BEV_RANGE, BEV_RESOLUTION, step)
            bev_acc += acc
            bev_cnt += cnt
        except Exception as e:
            print(f"  跳过 {cam}: {e}")

    # 平均颜色
    mask = bev_cnt > 0
    bev = np.zeros_like(bev_acc)
    bev[mask] = bev_acc[mask] / bev_cnt[mask][:, None]
    bev[~mask] = 0

    # 填充空洞
#    if fill_holes:
#        bev = fill_bev_holes(bev, max_distance=5)

    # 对比度增强
    if np.any(mask):
        vmin = np.percentile(bev[mask], 2)
        vmax = np.percentile(bev[mask], 98)
        bev = np.clip((bev - vmin) / (vmax - vmin + 1e-6), 0, 1)

    return bev

# ========== 主程序 ==========
if __name__ == "__main__":
    # 获取一个样本token（随便取一个）
    sample_data = load_json('v1.0-mini/sample_data.json')
    sample_token = None
    for sd in sample_data:
        if sd.get('is_key_frame') and sd.get('filename', '').startswith('samples/CAM_FRONT'):
            sample_token = sd['sample_token']
            break

    if sample_token is None:
        raise ValueError("未找到有效样本")

    print(f"生成BEV全景图，样本: {sample_token}")
    bev = generate_bev_panorama(sample_token, step=2, fill_holes=True)

    # 显示
    plt.figure(figsize=(8,8))
    plt.imshow(bev)
    plt.title("540° BEV Panorama (6 cameras)")
    plt.text(BEV_SIZE//2, 20, '⬆ Front', color='red', weight='bold', ha='center')
    plt.text(BEV_SIZE//2, BEV_SIZE-20, '⬇ Back', color='red', weight='bold', ha='center')
    plt.text(20, BEV_SIZE//2, '⬅ Left', color='red', weight='bold', va='center')
    plt.text(BEV_SIZE-20, BEV_SIZE//2, '➡ Right', color='red', weight='bold', va='center')
    plt.axis('off')
    plt.show()
