import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
from model import init_model_params, bev_forward, bev_backward, compute_losses, d_compute_losses

from train_test import  clip_gradients,cosine_annealing
# ============================
# 配置
# ============================
DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"  # 修改为你的路径
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
BEV_RANGE = (-50, 50)          # 米
BEV_RESOLUTION = 1.0           # 米/像素
BEV_SIZE = (int((BEV_RANGE[1] - BEV_RANGE[0]) / BEV_RESOLUTION),
            int((BEV_RANGE[1] - BEV_RANGE[0]) / BEV_RESOLUTION))
DEPTH_BINS = 41
DEPTH_RANGE = (4.0, 45.0)      # 米
NUM_CLASSES = 10
BATCH_SIZE = 1  # 先使用单样本训练

# ============================
# 工具函数
# ============================
def adam_update(params, grads, lr, step, m=None, v=None, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    递归更新嵌套参数字典，支持列表。
    params: 嵌套参数字典（可包含列表）
    grads: 与 params 结构相同的梯度
    step: 当前步数（从1开始）
    m, v: 动量缓存（与 params 结构相同），若为 None 则初始化
    返回: (params, m, v, step+1)
    """
    if m is None:
        m = {}
    if v is None:
        v = {}

    def _update(p, g, m_cur, v_cur, lr_scale):
        if isinstance(p, dict):
            # 确保 m_cur 和 v_cur 是字典
            if not isinstance(m_cur, dict):
                m_cur = {}
            if not isinstance(v_cur, dict):
                v_cur = {}
            for key in p:
                if key in g:
                    m_cur[key], v_cur[key] = _update(
                        p[key], g[key],
                        m_cur.get(key, {}),
                        v_cur.get(key, {}),
                        lr_scale
                    )
        elif isinstance(p, list):
            if not isinstance(g, list) or len(p) != len(g):
                raise ValueError("参数和梯度列表长度不匹配")
            # 确保 m_cur 和 v_cur 是列表
            if not isinstance(m_cur, list):
                m_cur = [None] * len(p)
            if not isinstance(v_cur, list):
                v_cur = [None] * len(p)
            for i in range(len(p)):
                m_cur[i], v_cur[i] = _update(
                    p[i], g[i],
                    m_cur[i] if i < len(m_cur) else None,
                    v_cur[i] if i < len(v_cur) else None,
                    lr_scale
                )
        else:
            # 数值更新
            # 确保 m_cur 和 v_cur 是数组
            if not isinstance(m_cur, np.ndarray):
                m_cur = np.zeros_like(p)
            if not isinstance(v_cur, np.ndarray):
                v_cur = np.zeros_like(p)
            m_cur = beta1 * m_cur + (1 - beta1) * g
            v_cur = beta2 * v_cur + (1 - beta2) * (g * g)
            m_hat = m_cur / (1 - beta1 ** step)
            v_hat = v_cur / (1 - beta2 ** step)
            p -= lr * lr_scale * m_hat / (np.sqrt(v_hat) + eps)
        return m_cur, v_cur

    _update(params, grads, m, v, 1.0)
    return params, m, v, step + 1
def clip_gradients(grads, max_norm=1.0):
    """对嵌套字典中的梯度进行裁剪（按全局范数）"""
    # 计算总范数
    total_norm = 0.0
    def accumulate_norm(g):
        nonlocal total_norm
        if isinstance(g, dict):
            for v in g.values():
                accumulate_norm(v)
        elif isinstance(g, list):
            for v in g:
                accumulate_norm(v)
        else:
            total_norm += np.sum(g ** 2)
    accumulate_norm(grads)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        def scale_grads(g):
            if isinstance(g, dict):
                for k in g:
                    scale_grads(g[k])
            elif isinstance(g, list):
                for i in range(len(g)):
                    scale_grads(g[i])
            else:
                g *= scale
        scale_grads(grads)
    return grads
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

def load_sample_data(sample_token,target_size=(256,704)):
    """加载一个样本的6张图像和标定"""
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
    # 获取该样本的关键帧数据
    sample_data = [sd for sd in sample_data_list if sd['sample_token'] == sample_token and sd['is_key_frame']]
    images = {}
    calibs = {}
    ego_pose_mat = None
    # 原始图像尺寸（nuScenes 标准）
    orig_h, orig_w = 900, 1600
    target_h, target_w = target_size
    scale_y = target_h / orig_h
    scale_x = target_w / orig_w
    for sd in sample_data:
        calib_token = sd['calibrated_sensor_token']
        if calib_token not in calib_token_to_sensor:
            continue
        sensor_token = calib_token_to_sensor[calib_token]['sensor_token']
        channel = sensor_token_to_channel.get(sensor_token)
        if channel not in CAMERAS:
            continue
        img_path = os.path.join(DATA_ROOT, sd['filename'])
        img = plt.imread(img_path)  # (H, W, 3) 0-1 float
        if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
            # 使用 skimage.transform.resize 或 cv2.resize
            from skimage.transform import resize
            img = resize(img, target_size, preserve_range=True, anti_aliasing=True)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        images[channel] = img
       # 获取内参并缩放
        intrin = calib_token_to_intrin.get(calib_token)
        if intrin is not None:
            intrin_scaled = intrin.copy()
            intrin_scaled[0, :] *= scale_x
            intrin_scaled[1, :] *= scale_y
        else:
            intrin_scaled = None
        R = quaternion_to_rotation_matrix(calib_token_to_sensor[calib_token]['rotation'])
        t = np.array(calib_token_to_sensor[calib_token]['translation'])
        T_sensor2ego = np.eye(4)
        T_sensor2ego[:3, :3] = R
        T_sensor2ego[:3, 3] = t
        calibs[channel] = {'intrin': intrin, 'extrin': T_sensor2ego}
        # 获取 ego_pose
        ego_pose = ego_pose_map[sd['ego_pose_token']]
        ego_pose_mat = np.eye(4)
        ego_pose_mat[:3, :3] = quaternion_to_rotation_matrix(ego_pose['rotation'])
        ego_pose_mat[:3, 3] = np.array(ego_pose['translation'])
    # 按顺序整理
    img_list = [images.get(cam) for cam in CAMERAS]
    calib_list = [calibs.get(cam) for cam in CAMERAS]
    return img_list, calib_list, ego_pose_mat

def build_geometry_indices(sample_token, model_params, H, W, Hf, Wf):
    """
    为每个相机计算 BEV 网格索引。
    返回: list of (D*Hf*Wf,) 数组，每个相机一个
    """
    # 获取标定
    _, calib_list, ego_pose_mat = load_sample_data(sample_token)
    # 预计算每个相机的视锥点云坐标
    depth_bins = DEPTH_BINS
    depth_range = DEPTH_RANGE
    bev_size = BEV_SIZE
    bev_res = BEV_RESOLUTION
    bev_range = BEV_RANGE
    indices_list = []
    for cam_idx, calib in enumerate(calib_list):
        if calib is None:
            indices_list.append(None)
            continue
        intrin = calib['intrin']
        extrin = calib['extrin']  # 相机→自车
        # 生成视锥点云 (D, Hf, Wf, 3) 在相机坐标系下
        # 假设图像特征图尺寸 Hf, Wf
        # 生成网格点
        u = np.linspace(0, W-1, Wf, dtype=np.float32)
        v = np.linspace(0, H-1, Hf, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        # 深度值
        d = np.linspace(depth_range[0], depth_range[1], depth_bins, dtype=np.float32)
        # 构建点云坐标 (D, Hf, Wf, 3)
        # 利用内参逆变换
        fx, fy = intrin[0,0], intrin[1,1]
        cx, cy = intrin[0,2], intrin[1,2]
        x_cam = (uu - cx) / fx
        y_cam = (vv - cy) / fy
        # 扩展深度维度
        z_cam = d[:, None, None]
        points_cam = np.stack([x_cam[None, ...] * z_cam,
                               y_cam[None, ...] * z_cam,
                               z_cam * np.ones_like(x_cam[None, ...])], axis=-1)  # (D, Hf, Wf, 3)
        # 转换到自车坐标系
        R = extrin[:3, :3]
        t = extrin[:3, 3]
        points_ego = np.einsum('ij,dhwk->dhwi', R, points_cam) + t  # (D, Hf, Wf, 3)
        # 只保留地面附近点 (z=0)
        # 但我们要将所有点投影到 BEV 网格，并记录网格索引
        x_ego = points_ego[..., 0]
        y_ego = points_ego[..., 1]
        # 计算网格索引
        gx = ((x_ego - bev_range[0]) / bev_res).astype(np.int32)
        gy = ((y_ego - bev_range[0]) / bev_res).astype(np.int32)
        # 过滤超出范围的点
        mask = (gx >= 0) & (gx < bev_size[0]) & (gy >= 0) & (gy < bev_size[1])
        # 将有效点映射到一维索引
        grid_idx = gx * bev_size[1] + gy
        grid_idx = grid_idx[mask]
        # 展平所有维度
        indices = grid_idx.flatten()
        indices_list.append(indices)
    return indices_list

def get_annotations(sample_token):
    sample_anns = load_json('v1.0-mini/sample_annotation.json')
    instances = load_json('v1.0-mini/instance.json')
    categories = load_json('v1.0-mini/category.json')
    inst_to_cat = {inst['token']: inst['category_token'] for inst in instances}
    cat_to_name = {cat['token']: cat['name'] for cat in categories}
    anns = []
    for ann in sample_anns:
        if ann['sample_token'] == sample_token:
            inst_token = ann['instance_token']
            if inst_token in inst_to_cat:
                cat_token = inst_to_cat[inst_token]
                cat_name = cat_to_name.get(cat_token, 'unknown')
                if cat_name.startswith('vehicle.'):
                    anns.append(ann)
    return anns

def generate_heatmap_gt(anns, ego_pose_mat, bev_size, sigma=3.0):
    heatmap_gt = np.zeros((1, NUM_CLASSES, bev_size[0], bev_size[1]), dtype=np.float32)
    # 高斯核
    def gaussian_2d(shape, center, sigma):
        y, x = np.ogrid[:shape[0], :shape[1]]
        x0, y0 = center
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    ego_pose_inv = np.linalg.inv(ego_pose_mat)
    R_global2ego = ego_pose_inv[:3, :3]
    t_global2ego = ego_pose_inv[:3, 3]
    for ann in anns:
        pos_global = np.array(ann['translation'])
        pos_ego = R_global2ego @ pos_global + t_global2ego
        x, y = pos_ego[1], pos_ego[0]
        if x < BEV_RANGE[0] or x >= BEV_RANGE[1] or y < BEV_RANGE[0] or y >= BEV_RANGE[1]:
            continue
        gx = int((x - BEV_RANGE[0]) / BEV_RESOLUTION)
        gy = int((y - BEV_RANGE[0]) / BEV_RESOLUTION)
        gx = bev_size[0] - 1 - gx
        gy = bev_size[1] - 1 - gy
        if 0 <= gx < bev_size[0] and 0 <= gy < bev_size[1]:
            # 放置高斯热图（只取类别0）
            heat = gaussian_2d((bev_size[0], bev_size[1]), (gx, gy), sigma)
            heatmap_gt[0, 0] += heat
    # 归一化到0-1
    heatmap_gt = np.clip(heatmap_gt, 0, 1)
    return heatmap_gt

def generate_reg_gt(anns, ego_pose_mat, bev_size):
    reg_gt = np.zeros((1, 8, bev_size[0], bev_size[1]), dtype=np.float32)
    ego_pose_inv = np.linalg.inv(ego_pose_mat)
    R_global2ego = ego_pose_inv[:3, :3]
    t_global2ego = ego_pose_inv[:3, 3]
    for ann in anns:
        pos_global = np.array(ann['translation'])
        pos_ego = R_global2ego @ pos_global + t_global2ego
        x, y = pos_ego[1], pos_ego[0]
        if x < BEV_RANGE[0] or x >= BEV_RANGE[1] or y < BEV_RANGE[0] or y >= BEV_RANGE[1]:
            continue
        gx = int((x - BEV_RANGE[0]) / BEV_RESOLUTION)
        gy = int((y - BEV_RANGE[0]) / BEV_RESOLUTION)
        gx = bev_size[0] - 1 - gx
        gy = bev_size[1] - 1 - gy
        if 0 <= gx < bev_size[0] and 0 <= gy < bev_size[1]:
            w, l, h = ann['size']
            quat = ann['rotation']
            yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                             1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
            reg_gt[0, 2, gx, gy] = w
            reg_gt[0, 3, gx, gy] = l
            reg_gt[0, 4, gx, gy] = h
            reg_gt[0, 5, gx, gy] = np.sin(yaw)
            reg_gt[0, 6, gx, gy] = np.cos(yaw)
            reg_gt[0, 7, gx, gy] = pos_ego[2]  # depth
    return reg_gt

def generate_depth_gt(sample_token, Hf, Wf):
    # 暂时使用随机深度
    depth_gt = np.random.randint(0, DEPTH_BINS, size=(1, Hf, Wf)).astype(np.int32)
    return depth_gt
def generate_depth_gt_(sample_token, data_root, Hf, Wf, depth_bins=41, depth_range=(4.0, 45.0)):
    """
    从激光雷达点云生成深度真值图，仅前摄像头
    """
    # 加载 JSON
    sample_data_list = load_json('v1.0-mini/sample_data.json')
    sensors = load_json('v1.0-mini/sensor.json')
    calib_sensors = load_json('v1.0-mini/calibrated_sensor.json')
    ego_poses = load_json('v1.0-mini/ego_pose.json')

    # 构建映射
    sensor_token_to_channel = {s['token']: s['channel'] for s in sensors}
    calib_token_to_intrin = {}
    calib_token_to_extrin = {}
    calib_token_to_sensor_token = {}
    for cs in calib_sensors:
        token = cs['token']
        if 'camera_intrinsic' in cs:
            calib_token_to_intrin[token] = np.array(cs['camera_intrinsic'])
        calib_token_to_extrin[token] = {
            'rotation': cs['rotation'],
            'translation': cs['translation']
        }
        calib_token_to_sensor_token[token] = cs['sensor_token']
    ego_pose_map = {ep['token']: ep for ep in ego_poses}

    # 查找 LIDAR_TOP 和前摄像头
    lidar_sd = None
    cam_sd = None
    for sd in sample_data_list:
        if sd['sample_token'] != sample_token or not sd['is_key_frame']:
            continue
        calib_token = sd['calibrated_sensor_token']
        sensor_token = calib_token_to_sensor_token.get(calib_token)
        if sensor_token is None:
            continue
        channel = sensor_token_to_channel.get(sensor_token)
        if channel == 'LIDAR_TOP':
            lidar_sd = sd
        elif channel == 'CAM_FRONT':
            cam_sd = sd
    if lidar_sd is None or cam_sd is None:
        return np.full((1, Hf, Wf), -1, dtype=np.int32)

    # 加载点云
    lidar_path = os.path.join(data_root, lidar_sd['filename'])
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]

    # 获取标定
    cam_calib_token = cam_sd['calibrated_sensor_token']
    intrin = calib_token_to_intrin.get(cam_calib_token)
    if intrin is None:
        return np.full((1, Hf, Wf), -1, dtype=np.int32)

    extrin_data = calib_token_to_extrin[cam_calib_token]
    R_sensor2ego = quaternion_to_rotation_matrix(extrin_data['rotation'])
    t_sensor2ego = np.array(extrin_data['translation'])
    T_sensor2ego = np.eye(4)
    T_sensor2ego[:3, :3] = R_sensor2ego
    T_sensor2ego[:3, 3] = t_sensor2ego
    T_ego2sensor = np.linalg.inv(T_sensor2ego)

    ego_pose_token = cam_sd['ego_pose_token']
    ego_pose = ego_pose_map[ego_pose_token]
    R_ego2global = quaternion_to_rotation_matrix(ego_pose['rotation'])
    t_ego2global = np.array(ego_pose['translation'])
    T_ego2global = np.eye(4)
    T_ego2global[:3, :3] = R_ego2global
    T_ego2global[:3, 3] = t_ego2global
    T_global2ego = np.linalg.inv(T_ego2global)

    # 全局→自车→相机
    points_ego = (T_global2ego[:3, :3] @ points.T).T + T_global2ego[:3, 3]
    points_cam = (T_ego2sensor[:3, :3] @ points_ego.T).T + T_ego2sensor[:3, 3]
    valid_mask = points_cam[:, 2] > 0
    points_cam = points_cam[valid_mask]
    if points_cam.shape[0] == 0:
        return np.full((1, Hf, Wf), -1, dtype=np.int32)

    # 投影到图像
    fx, fy = intrin[0,0], intrin[1,1]
    cx, cy = intrin[0,2], intrin[1,2]
    u = (points_cam[:, 0] * fx) / points_cam[:, 2] + cx
    v = (points_cam[:, 1] * fy) / points_cam[:, 2] + cy
    depth = points_cam[:, 2]

    img_h, img_w = 900, 1600
    mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[mask].astype(np.int32)
    v = v[mask].astype(np.int32)
    depth = depth[mask]
    if len(depth) == 0:
        return np.full((1, Hf, Wf), -1, dtype=np.int32)

    # 离散化深度
    d_min, d_max = depth_range
    depth_bins_edges = np.linspace(d_min, d_max, depth_bins+1)
    depth_indices = np.clip(np.digitize(depth, depth_bins_edges) - 1, 0, depth_bins-1)

    # 创建深度图
    depth_map = np.full((img_h, img_w), -1, dtype=np.int32)
    depth_map[v, u] = depth_indices

    # 下采样到特征图尺寸（最近邻）
    from scipy.ndimage import zoom
    # 计算缩放因子
    scale_h = Hf / img_h
    scale_w = Wf / img_w
    depth_gt = zoom(depth_map.astype(np.float32), (scale_h, scale_w), order=0, mode='nearest')
    depth_gt = depth_gt.astype(np.int32)
    depth_gt = depth_gt[None, :, :]  # (1, Hf, Wf)
    return depth_gt
# ============================
# 训练主循环
# ============================
def main():
    # 初始化模型参数
    model_params = init_model_params(
        backbone_in=3,
        backbone_out=64,
        depth_bins=DEPTH_BINS,
        context_channels=64,
        bev_channels=64,
        bev_shape=BEV_SIZE,
        num_classes=NUM_CLASSES
    )

    # 获取样本列表
    samples = load_json('v1.0-mini/sample.json')
    sample_tokens = [s['token'] for s in samples[:20]]  # 使用前20个样本测试

    # 确定特征图尺寸（由模型决定）
    H, W = 256, 704  # 输入图像尺寸（需与模型一致）
    Hf, Wf = H // 32, W // 32  # 8, 22

    # 预计算几何索引（耗时，可缓存）
    geom_indices_cache = {}
    for token in tqdm(sample_tokens, desc="Building geometry indices"):
        geom_indices_cache[token] = [build_geometry_indices(token, model_params, H, W, Hf, Wf)]

    # 超参数
    lr_init = 1e-3
    epochs = 10
    hm_weight = 0.1
    reg_weight = 1.0
    depth_weight = 1.0

    step =1
    m, v = None, None

    for epoch in range(epochs):
        total_loss = 0.0
        lr = cosine_annealing(epoch,epochs,lr_init=lr_init,lr_min=1e-6) 
#        for sample_token in tqdm(sample_tokens, desc=f"Epoch {epoch+1}/{epochs}"):
        for sample_token in tqdm([sample_tokens[2]], desc=f"Epoch {epoch+1}/{epochs}"):
            # 1. 加载图像和标定
            img_list, calib_list, ego_pose_mat = load_sample_data(sample_token)
            # 转换为 (1, 6, 3, H, W) 格式
            images = np.stack([img.transpose(2,0,1) for img in img_list if img is not None], axis=0)  # (6,3,H,W)
            # 添加 batch 维度
            images = images[None, ...]  # (1,6,3,H,W)
            # 如果某个相机缺失，用零填充（但 sample 中应都有）
            # 调整顺序与 CAMERAS 一致
            # 略

            # 2. 获取几何索引
            geom_indices = geom_indices_cache[sample_token]
            # 3. 前向
            heatmap, reg, depth_logits_list, bev_feat, caches = bev_forward(
                images, geom_indices, BEV_SIZE, model_params
            )

            # 4. 获取标注
            anns = get_annotations(sample_token)

            # 5. 生成 GT
            heatmap_gt = generate_heatmap_gt(anns, ego_pose_mat, BEV_SIZE, sigma=3.0)
            reg_gt = generate_reg_gt(anns, ego_pose_mat, BEV_SIZE)
            depth_gt = generate_depth_gt(sample_token, Hf, Wf)
            import pdb;pdb.set_trace()
            # 6. 堆叠深度 logits（取第一个相机，仅前摄像头）
            depth_logits_batch = depth_logits_list[0][None, ...]  # (1, D, Hf, Wf)

            # 7. 损失计算（加权）
            losses, loss_caches = compute_losses(
                heatmap, reg, heatmap_gt, reg_gt,
                depth_logits_batch, depth_gt,
                hm_weight=hm_weight, reg_weight=reg_weight, depth_weight=depth_weight
            )
            total_loss += losses['total_loss']

            # 8. 梯度
            dheatmap, dreg, ddepth = d_compute_losses(
                heatmap, reg, heatmap_gt, reg_gt,
                depth_logits_batch, depth_gt, loss_caches,hm_weight=hm_weight,reg_weight = reg_weight,depth_weight=1.0)

            # 9. 构建深度梯度列表（每个相机）
            ddepth_per_cam = [None] * 6
            ddepth_per_cam[0] = ddepth  # 仅前摄像头有深度梯度

            # 10. 反向传播
            grads = bev_backward(
                dheatmap, dreg, ddepth_per_cam,
                caches, model_params, geom_indices, BEV_SIZE
            )

            # 11. 更新
            model_params, m, v, step = adam_update(model_params, grads, lr, step, m, v)
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch:3d}, total: {losses['total_loss']:.6f},hm:{losses['loss_heatmap']:.6f},reg:{losses['loss_reg']:.6f},depth:{losses['loss_depth']:.6f}")

#        avg_loss = total_loss / len(sample_tokens)
#        print(f"Epoch {epoch+1}/{epochs}, avg loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
