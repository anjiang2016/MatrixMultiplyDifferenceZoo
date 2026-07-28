# bev_stitch_range.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings("ignore")

# ========== 配置 ==========
DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"
ORDER = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
         'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
BEV_RANGE = (-50, 50)          # 扩大到 ±50 米
BEV_RESOLUTION = 0.1           # 分辨率 0.1 米/像素，避免网格太大
BEV_SIZE = int((BEV_RANGE[1] - BEV_RANGE[0]) / BEV_RESOLUTION)  # 1000

# ========== 工具函数 ==========
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

def load_sample_data(sample_token):
    sensors = load_json('v1.0-mini/sensor.json')
    sensor_token_to_channel = {s['token']: s['channel'] for s in sensors}
    calib_sensors = load_json('v1.0-mini/calibrated_sensor.json')
    calib_token_to_sensor = {}
    calib_token_to_intrin = {}
    for cs in calib_sensors:
        calib_token_to_sensor[cs['token']] = {
            'sensor_token': cs['sensor_token'],
            'rotation': cs['rotation'],
            'translation': cs['translation']
        }
        if 'camera_intrinsic' in cs:
            calib_token_to_intrin[cs['token']] = np.array(cs['camera_intrinsic'])
    ego_poses = load_json('v1.0-mini/ego_pose.json')
    ego_pose_map = {ep['token']: ep for ep in ego_poses}
    sample_data_list = load_json('v1.0-mini/sample_data.json')
    sample_data = [sd for sd in sample_data_list if sd['sample_token'] == sample_token and sd['is_key_frame']]
    images = {}
    calibs = {}
    for sd in sample_data:
        calib_token = sd['calibrated_sensor_token']
        if calib_token not in calib_token_to_sensor:
            continue
        sensor_token = calib_token_to_sensor[calib_token]['sensor_token']
        channel = sensor_token_to_channel.get(sensor_token)
        if channel not in ORDER:
            continue
        img_path = os.path.join(DATA_ROOT, sd['filename'])
        images[channel] = mpimg.imread(img_path)
        intrin = calib_token_to_intrin.get(calib_token)
        # 自车→传感器
        R_ego2sensor = quaternion_to_rotation_matrix(calib_token_to_sensor[calib_token]['rotation'])
        t_ego2sensor = np.array(calib_token_to_sensor[calib_token]['translation'])
        T_ego2sensor = np.eye(4)
        T_ego2sensor[:3, :3] = R_ego2sensor
        T_ego2sensor[:3, 3] = t_ego2sensor
        T_sensor2ego = np.linalg.inv(T_ego2sensor)
        calibs[channel] = {'intrin': intrin, 'extrin': T_ego2sensor}
    img_list = [images.get(cam) for cam in ORDER]
    calib_list = [calibs.get(cam) for cam in ORDER]
    return img_list, calib_list
def project_camera_to_bev(img, intrin, extrin, bev_range, bev_res, step=2, flip_y=True):
    """
    将单张相机图像投影到 BEV 网格（兼容原函数接口）
    参数:
        img: (H, W, 3) 图像
        intrin: (3,3) 内参矩阵
        extrin: (4,4) 相机→自车变换矩阵（即 calibrated_sensor 的 rotation + translation 构成的 4x4 矩阵）
        bev_range: (min, max) BEV 范围（米）
        bev_res: 分辨率（米/像素）
        step: 采样步长（默认2）
        flip_y: 是否翻转 y 轴使左侧在左（默认 True）
    返回:
        grid_x, grid_y, colors: 有效像素的网格索引和颜色值
    """
    # 从 extrin 提取旋转和平移（传感器→自车）
    R_cam2ego = extrin[:3, :3]
    t_cam2ego = extrin[:3, 3]

    # 图像归一化处理
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

    H, W, _ = img.shape
    fx, fy = intrin[0,0], intrin[1,1]
    cx, cy = intrin[0,2], intrin[1,2]

    # 生成像素网格
    uu, vv = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    u = uu.ravel()
    v = vv.ravel()

    # 归一化射线
    x_c = (u - cx) / fx
    y_c = (v - cy) / fy
    rays = np.stack([x_c, y_c, np.ones_like(u)], axis=1)

    # 旋转到自车坐标系
    d_ego = (R_cam2ego @ rays.T).T

    # 过滤：只保留指向地面的射线 (d_z < 0)
    valid = d_ego[:, 2] < 0
    if not np.any(valid):
        return np.array([]), np.array([]), np.array([])

    d_ego = d_ego[valid]
    rays = rays[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    # 地面交点：λ = -t_z / d_z
    lam = -t_cam2ego[2] / d_ego[:, 2]
    P_ego = lam[:, None] * d_ego + t_cam2ego

    x_ego = P_ego[:, 0]
    y_ego = P_ego[:, 1]

    # 过滤 BEV 范围内
    bev_size = int((bev_range[1] - bev_range[0]) / bev_res)
    idx = (x_ego >= bev_range[0]) & (x_ego < bev_range[1]) & (y_ego >= bev_range[0]) & (y_ego < bev_range[1])
    if not np.any(idx):
        return np.array([]), np.array([]), np.array([])

    x_ego = x_ego[idx]
    y_ego = y_ego[idx]
    u_valid = u_valid[idx]
    v_valid = v_valid[idx]

    # 网格索引
    grid_x = ((x_ego - bev_range[0]) / bev_res).astype(np.int32)
    grid_y = ((y_ego - bev_range[0]) / bev_res).astype(np.int32)

    # 翻转 x 使前方在上，翻转 y 使左侧在左
    grid_x = bev_size - 1 - grid_x
    if flip_y:
        grid_y = bev_size - 1 - grid_y

    grid_x = np.clip(grid_x, 0, bev_size - 1)
    grid_y = np.clip(grid_y, 0, bev_size - 1)

    colors = img[v_valid, u_valid]
    return grid_x, grid_y, colors
def project_camera_to_bev_(img, intrin, extrin, bev_range, bev_res):
    H, W, _ = img.shape
    step = 2
    uu, vv = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    u = uu.ravel()
    v = vv.ravel()
    intrin_inv = np.linalg.inv(intrin)
    pix_hom = np.stack([u, v, np.ones_like(u)], axis=1)
    rays = (intrin_inv @ pix_hom.T).T   # 相机坐标系下的方向 (x_c, y_c, 1)
    R = extrin[:3, :3]
    t = extrin[:3, 3]
    # 计算与地面 z=0 的交点
    A = R[2, 0]*rays[:,0] + R[2,1]*rays[:,1] + R[2,2]*rays[:,2]
    valid = np.abs(A) > 1e-8
    if not np.any(valid):
        return np.array([]), np.array([]), np.array([])
    u_valid = u[valid]
    v_valid = v[valid]
    rays_valid = rays[valid]
    A_valid = A[valid]
    lambda_ = -t[2] / A_valid
    P_ego = lambda_[:, None] * (rays_valid @ R.T) + t
    x = P_ego[:, 0]
    y = P_ego[:, 1]
    # 交换 x 和 y（根据之前的调试）
    x, y = y, x
    # 过滤 BEV 范围
    idx = (x >= bev_range[0]) & (x < bev_range[1]) & (y >= bev_range[0]) & (y < bev_range[1])
    x = x[idx]
    y = y[idx]
    u_valid = u_valid[idx]
    v_valid = v_valid[idx]
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    # 网格索引
    grid_x = ((x - bev_range[0]) / bev_res).astype(np.int)
    grid_x = BEV_SIZE - 1 - grid_x   # 翻转 X 轴
    grid_y = ((y - bev_range[0]) / bev_res).astype(np.int)
    grid_x = np.clip(grid_x, 0, BEV_SIZE-1)
    grid_y = np.clip(grid_y, 0, BEV_SIZE-1)
    colors = img[v_valid, u_valid]
    return grid_x, grid_y, colors
if __name__ == "__main__":
    import os
    # 创建输出目录
    output_dir = "bev_stitch_output"
    os.makedirs(output_dir, exist_ok=True)

    samples = load_json('v1.0-mini/sample.json')
    sample_tokens = [s['token'] for s in samples]
    total = len(sample_tokens)
    print(f"共 {total} 个样本，开始生成 BEV 序列...")

    for idx, sample_token in enumerate(sample_tokens):
        print(f"处理样本 {idx+1}/{total}: {sample_token}")
        try:
            img_list, calib_list = load_sample_data(sample_token)
            bev = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.float32)
            count = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint16)

            for cam_idx, (img, calib) in enumerate(zip(img_list, calib_list)):
                if img is None or calib is None:
                    continue
                intrin = calib['intrin']
                extrin = calib['extrin']
                if intrin is None:
                    continue
                grid_x, grid_y, colors = project_camera_to_bev(img, intrin, extrin, BEV_RANGE, BEV_RESOLUTION)
                if len(grid_x) > 0:
                    np.add.at(bev, (grid_x, grid_y), colors)
                    np.add.at(count, (grid_x, grid_y), 1)

            mask = count > 0
            bev[mask] = bev[mask] / count[mask][:, None]
            bev[~mask] = [0, 0, 0]

            if np.any(mask):
                vmin = np.percentile(bev[mask], 2)
                vmax = np.percentile(bev[mask], 98)
                bev = np.clip((bev - vmin) / (vmax - vmin + 1e-6), 0, 1)

            # 保存图片（使用 matplotlib 的 imsave，避免额外显示）
            save_path = os.path.join(output_dir, f"bev_{idx+1:04d}_{sample_token[:8]}.png")
            plt.imsave(save_path, bev)
            if (idx + 1) % 10 == 0 or idx == total - 1:
                print(f"  已保存 {idx+1} 张图片")
        except Exception as e:
            print(f"  处理样本 {sample_token} 时出错: {e}")
            continue

    print("全部完成！")
