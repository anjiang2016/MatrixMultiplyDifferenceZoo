import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
from scipy.ndimage import distance_transform_edt, gaussian_filter

def fill_bev_holes(bev, max_distance=5):
    """
    用最近邻插值填充 BEV 图中的空洞。
    参数:
        bev: (H, W, 3) 浮点数数组，背景为黑色 (0,0,0)
        max_distance: 最大插值距离（像素），超过此距离保持黑色
    返回:
        filled: (H, W, 3) 填充后的图像
    """
    # 创建有效像素掩码（非背景）
    mask = (bev.sum(axis=-1) > 0).astype(np.uint8)
    import pdb;pdb.set_trace()    
    # 如果没有空洞，直接返回
    if np.all(mask):
        return bev
    
    # 距离变换：每个像素到最近有效像素的距离
    dist, idx = distance_transform_edt(mask, return_indices=True)
    
    # 对每个通道分别插值
    filled = np.zeros_like(bev)
    for c in range(3):
        # 取有效像素的颜色值
        valid_vals = bev[:, :, c]
        # 用最近有效像素的值填充
        filled[:, :, c] = valid_vals[idx[0], idx[1]]
    
    # 将距离超过 max_distance 的区域置回黑色（避免远处误填充）
    filled[dist > max_distance] = 0
    
    return filled

def load_calibration(data_root):
    """加载前摄像头标定数据，返回内参、R_cam2ego、t_cam2ego"""
    with open(os.path.join(data_root, 'v1.0-mini/sensor.json'), 'r') as f:
        sensors = json.load(f)
    sensor_token_to_channel = {s['token']: s['channel'] for s in sensors}

    with open(os.path.join(data_root, 'v1.0-mini/calibrated_sensor.json'), 'r') as f:
        calibs = json.load(f)

    for cs in calibs:
        sensor_token = cs['sensor_token']
        if sensor_token not in sensor_token_to_channel:
            continue
        if sensor_token_to_channel[sensor_token] != 'CAM_FRONT':
            continue
        intrin = np.array(cs['camera_intrinsic'])
        # 四元数转旋转矩阵（传感器→自车）
        q = cs['rotation']
        w, x, y, z = q
        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])
        t = np.array(cs['translation'])
        return intrin, R, t
    raise ValueError("未找到 CAM_FRONT 标定")
def project_cam_front_to_bev(image, intrin, R_cam2ego, t_cam2ego,
                             bev_range=(-20, 20), bev_res=0.1, step=2,
                             fill_holes=True):
    """
    将前摄像头图像投影到 BEV 俯视图，并可选填充空洞。
    """
    H, W, _ = image.shape
    fx, fy = intrin[0,0], intrin[1,1]
    cx, cy = intrin[0,2], intrin[1,2]

    bev_size = int((bev_range[1] - bev_range[0]) / bev_res)
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.float32)
    count = np.zeros((bev_size, bev_size), dtype=np.uint16)

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
            gx = bev_size - 1 - gx
            if 0 <= gx < bev_size and 0 <= gy < bev_size:
                bev[gx, gy] += image[v, u]
                count[gx, gy] += 1

    # 平均颜色
    mask = count > 0
    bev[mask] = bev[mask] / count[mask][:, None]
    bev[~mask] = 0

    # 填充空洞
    if fill_holes:
        bev = fill_bev_holes(bev, max_distance=5)

    # 对比度增强
    if np.any(mask):
        vmin = np.percentile(bev[mask], 2)
        vmax = np.percentile(bev[mask], 98)
        bev = np.clip((bev - vmin) / (vmax - vmin + 1e-6), 0, 1)

    return bev


# ========== 使用示例 ==========
if __name__ == "__main__":
    data_root = "/Users/zhaomingming/data_sets/v1.0-mini"

    # 1. 获取标定
    intrin, R_cam2ego, t_cam2ego = load_calibration(data_root)

    # 2. 加载一张前摄像头图像（从sample_data中找一个关键帧）
    with open(os.path.join(data_root, 'v1.0-mini/sample_data.json'), 'r') as f:
        sample_data = json.load(f)
    img_path = None
    for sd in sample_data:
        if sd.get('is_key_frame') and sd.get('filename', '').startswith('samples/CAM_FRONT'):
            img_path = os.path.join(data_root, sd['filename'])
            break
    if img_path is None:
        raise ValueError("未找到前摄像头关键帧图像")

    image = mpimg.imread(img_path)  # (900, 1600, 3)

    # 3. 生成BEV
    bev = project_cam_front_to_bev(image, intrin, R_cam2ego, t_cam2ego,
                                   bev_range=(-20, 20), bev_res=0.1, step=3)

    # 4. 显示
    plt.figure(figsize=(8, 8))
    plt.imshow(bev)
    plt.title("Front Camera BEV")
    plt.text(bev.shape[0]//2, 20, '⬆ Front', color='red', weight='bold', ha='center')
    plt.text(bev.shape[0]//2, bev.shape[0]-20, '⬇ Back', color='red', weight='bold', ha='center')
    plt.text(20, bev.shape[0]//2, '⬅ Left', color='red', weight='bold', va='center')
    plt.text(bev.shape[0]-20, bev.shape[0]//2, '➡ Right', color='red', weight='bold', va='center')
    plt.axis('off')
    plt.show()
