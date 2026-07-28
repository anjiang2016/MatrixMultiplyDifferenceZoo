import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
from scipy.ndimage import distance_transform_edt

DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT','CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
BEV_RANGE = (-20, 20)      # 米
BEV_RESOLUTION = 0.1       # 米/像素
BEV_SIZE = int((BEV_RANGE[1] - BEV_RANGE[0]) / BEV_RESOLUTION)

# ========== 工具函数（与之前相同） ==========
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
            d_ego[2]-=0.0
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
            gy = bev_size - 1 - gy
            if 0 <= gx < bev_size and 0 <= gy < bev_size:
                bev_acc[gx, gy] += image[v, u]
                bev_cnt[gx, gy] += 1
    return bev_acc, bev_cnt

def fill_bev_holes(bev, max_distance=5):
    mask = (bev.sum(axis=-1) > 0).astype(np.uint8)
    if np.sum(mask) == 0:
        return bev
    holes = (mask == 0).astype(np.uint8)
    dist, idx = distance_transform_edt(holes, return_indices=True)
    filled = np.zeros_like(bev)
    for c in range(3):
        filled[:, :, c] = bev[idx[0], idx[1], c]
    filled[dist > max_distance] = 0
    return filled
def generate_bev_panorama(sample_token, step=2, fill_holes=True):
    bev_acc = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.float32)
    bev_cnt = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint16)

    for cam in CAMERAS:
        try:
            intrin, R_cam2ego, t_cam2ego = get_camera_calib(cam)
            img_path = get_sample_image(cam, sample_token)
            image = mpimg.imread(img_path)
            acc, cnt = project_cam_to_bev(image, intrin, R_cam2ego, t_cam2ego,
                                          BEV_RANGE, BEV_RESOLUTION, step)
            bev_acc += acc
            bev_cnt += cnt
        except Exception as e:
            print(f"  警告: {cam} 在样本 {sample_token[:8]} 中缺失: {e}")

    mask = bev_cnt > 0
    valid_pixels = np.sum(mask)
    print(f"有效像素数: {valid_pixels} / {BEV_SIZE*BEV_SIZE}")

    if valid_pixels == 0:
        print("警告: 未生成任何有效像素，返回全黑图像")
        return np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.float32)

    bev = np.zeros_like(bev_acc)
    bev[mask] = bev_acc[mask] / bev_cnt[mask][:, None]
    bev[~mask] = 0
    if fill_holes:
        bev = fill_bev_holes(bev, max_distance=5)

    # 对比度增强（仅在有效像素值范围足够大时进行）
    if np.any(mask):
        vmin = np.percentile(bev[mask], 2)
        vmax = np.percentile(bev[mask], 98)
        # 如果范围太小，跳过增强，直接返回原始值
        if vmax - vmin > 0.01:
            bev = np.clip((bev - vmin) / (vmax - vmin + 1e-6), 0, 1)
        else:
            print(f"  跳过对比度增强（范围过小: {vmax-vmin:.4f}）")

    return bev

def draw_vehicle(ax, bev_size, car_length=4.0, car_width=2.0, res=0.1):
    half_l = int((car_length/2) / res)
    half_w = int((car_width/2) / res)
    center_x = bev_size // 2
    center_y = bev_size // 2
    rect = plt.Rectangle((center_y - half_w, center_x - half_l),
                         2*half_w, 2*half_l,
                         linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    ax.arrow(center_y, center_x, 0, -half_l*0.8,
             head_width=8, head_length=8, fc='white', ec='white')

def get_all_sample_tokens():
    samples = load_json('v1.0-mini/sample.json')
    return [s['token'] for s in samples]

# ========== 主程序：顺序读取所有样本并保存 BEV ==========
def main():
    sample_tokens = get_all_sample_tokens()
    print(f"共找到 {len(sample_tokens)} 个样本")

    # 创建保存目录
    save_dir = "bev_output"
    os.makedirs(save_dir, exist_ok=True)

    for idx, token in enumerate(sample_tokens):
        print(f"\n处理样本 {idx+1}/{len(sample_tokens)}: {token}")

        # 生成 BEV 全景图
        bev = generate_bev_panorama(token, step=4, fill_holes=True)
        # 保存为图片（带方向标注和车辆轮廓）
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(bev)
        ax.set_title(f"BEV {idx+1}")
        h, w = bev.shape[0], bev.shape[1]
        ax.text(w//2, 20, '⬆ Front', color='cyan', weight='bold', ha='center', fontsize=14)
        ax.text(w//2, h-20, '⬇ Back', color='cyan', weight='bold', ha='center', fontsize=14)
        ax.text(20, h//2, '⬅ Left', color='cyan', weight='bold', va='center', fontsize=14)
        ax.text(w-20, h//2, '➡ Right', color='cyan', weight='bold', va='center', fontsize=14)
        draw_vehicle(ax, BEV_SIZE)
        ax.axis('off')
        plt.tight_layout()
        # 保存图片
        save_path = os.path.join(save_dir, f"bev_{idx+1:04d}_{token[:8]}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  已保存: {save_path}")

    print("\n全部完成！")

if __name__ == "__main__":
    main()
