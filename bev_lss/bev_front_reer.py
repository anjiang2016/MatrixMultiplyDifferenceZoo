import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
from scipy.ndimage import distance_transform_edt

DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"
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
    """根据 channel 获取标定（内参、R_cam2ego、t_cam2ego）"""
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
        R_cam2ego = quat_to_rot(cs['rotation'])
        t_cam2ego = np.array(cs['translation'])
        return intrin, R_cam2ego, t_cam2ego
    raise ValueError(f"未找到 {channel} 标定")

def get_sample_image(channel, sample_token=None):
    sample_data = load_json('v1.0-mini/sample_data.json')
    
    # 预加载 sensor 和 calibrated_sensor 映射（可在外部缓存）
    sensors = load_json('v1.0-mini/sensor.json')
    sensor_token_to_channel = {s['token']: s['channel'] for s in sensors}
    calib_sensors = load_json('v1.0-mini/calibrated_sensor.json')
    calib_token_to_channel = {}
    for cs in calib_sensors:
        sensor_token = cs['sensor_token']
        if sensor_token in sensor_token_to_channel:
            calib_token_to_channel[cs['token']] = sensor_token_to_channel[sensor_token]
    
    if sample_token is None:
        # 找任意一个包含该 channel 关键帧的样本
        for sd in sample_data:
            if sd.get('is_key_frame'):
                calib_token = sd['calibrated_sensor_token']
                if calib_token in calib_token_to_channel and calib_token_to_channel[calib_token] == channel:
                    sample_token = sd['sample_token']
                    break
        if sample_token is None:
            raise ValueError(f"未找到 {channel} 的任何关键帧")
    
    # 在指定样本中查找该 channel 的关键帧
    for sd in sample_data:
        if sd['sample_token'] == sample_token and sd.get('is_key_frame'):
            calib_token = sd['calibrated_sensor_token']
            if calib_token in calib_token_to_channel and calib_token_to_channel[calib_token] == channel:
                return os.path.join(DATA_ROOT, sd['filename'])
    
    raise ValueError(f"未找到 {channel} 在样本 {sample_token} 中的图像")
def project_cam_to_bev(image, intrin, R_cam2ego, t_cam2ego,
                       bev_range, bev_res, step=2):
    """单相机投影到 BEV，返回 (bev, mask)"""
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
            d_ego +=0.0
            if d_ego[2] >= 0:
                continue
            lam = -t_cam2ego[2] / d_ego[2]
            P_ego = lam * d_ego + t_cam2ego
            x_ego, y_ego = P_ego[0], P_ego[1]
            if x_ego < bev_range[0] or x_ego >= bev_range[1] or y_ego < bev_range[0] or y_ego >= bev_range[1]:
                continue
            gx = int((x_ego - bev_range[0]) / bev_res)
            gy = int((y_ego - bev_range[0]) / bev_res)
            gx = bev_size - 1 - gx
            gy = bev_size  -1 - gy
            if 0 <= gx < bev_size and 0 <= gy < bev_size:
                bev_acc[gx, gy] += image[v, u]
                bev_cnt[gx, gy] += 1

    mask = bev_cnt > 0
    bev = np.zeros_like(bev_acc)
    bev[mask] = bev_acc[mask] / bev_cnt[mask][:, None]
    bev[~mask] = 0
    return bev, mask

def fill_bev_holes(bev, max_distance=5):
    # 有效像素掩码
    mask = (bev.sum(axis=-1) > 0).astype(np.uint8)
    if np.sum(mask) == 0:
        return bev

    # 空洞掩码（1 表示空洞）
    holes = (mask == 0).astype(np.uint8)

    # 计算每个空洞到最近有效像素的距离和索引
    from scipy.ndimage import distance_transform_edt
    dist, idx = distance_transform_edt(holes, return_indices=True)

    # 用最近有效像素的颜色填充空洞
    filled = np.zeros_like(bev)
    for c in range(3):
        filled[:, :, c] = bev[idx[0], idx[1], c]

    # 超过最大距离的保持黑色
    filled[dist > max_distance] = 0
    return filled
def generate_single_bev(camera_name, sample_token=None, step=2, fill_holes=True):
    """生成单个相机的 BEV 图"""
    print(f"生成 {camera_name} BEV...")
    intrin, R_cam2ego, t_cam2ego = get_camera_calib(camera_name)
    img_path = get_sample_image(camera_name, sample_token)
    print(img_path)
    image = mpimg.imread(img_path)
    bev, mask = project_cam_to_bev(image, intrin, R_cam2ego, t_cam2ego,
                                   BEV_RANGE, BEV_RESOLUTION, step)
    if fill_holes:
        bev = fill_bev_holes(bev, max_distance=5)
    # 对比度增强
    if np.any(mask):
        vmin = np.percentile(bev[mask], 2)
        vmax = np.percentile(bev[mask], 98)
        bev = np.clip((bev - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return bev

def main():
    # 获取一个样本 token（只要包含 CAM_FRONT 的关键帧）
    sample_data = load_json('v1.0-mini/sample_data.json')
    sample_token = None
    for sd in sample_data:
        if sd.get('is_key_frame') and sd.get('filename', '').startswith('samples/CAM_FRONT'):
            sample_token = sd['sample_token']
            break
    if sample_token is None:
        raise ValueError("未找到有效样本")

    # 生成前、后 BEV
    bev_front = generate_single_bev('CAM_FRONT_LEFT', sample_token, step=2, fill_holes=True)
    bev_back  = generate_single_bev('CAM_FRONT_RIGHT',  sample_token, step=2, fill_holes=True)

    # 并排显示
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    titles = ['Front Camera BEV', 'Back Camera BEV']

    for ax, bev, title in zip(axes, [bev_front, bev_back], titles):
        ax.imshow(bev)
        ax.set_title(title)
        ax.axis('off')
        # 方向标注
        h, w = bev.shape[0], bev.shape[1]
        ax.text(w//2, 20, '⬆ Front' if 'Front' in title else '⬆ Back', 
                color='red', weight='bold', ha='center')
        ax.text(w//2, h-20, '⬇ Back' if 'Front' in title else '⬇ Front', 
                color='red', weight='bold', ha='center')
        ax.text(20, h//2, '⬅ Left', color='red', weight='bold', va='center')
        ax.text(w-20, h//2, '➡ Right', color='red', weight='bold', va='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
