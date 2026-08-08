import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
from model import init_model_params, bev_forward, bev_backward, compute_losses, d_compute_losses
from train_test import  clip_gradients,cosine_annealing
from generate_heatmap import generate_inline_html,save_all_class_heatmaps
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funcs import sigmoid
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
BATCH_SIZE = 1  # 先使用单样本训练

class_mapping = {
    'movable_object.barrier': 0,
    'human.pedestrian.adult': 1,
    'movable_object.trafficcone': 2,
    'vehicle.car': 3,
    'vehicle.truck': 4,
    'human.pedestrian.construction_worker': 5,
    'vehicle.motorcycle': 6,
    'vehicle.construction': 7,
    'vehicle.bicycle': 8,
    'movable_object.pushable_pullable': 9,
    'vehicle.bus.rigid': 10,
}
NUM_CLASSES = len(class_mapping)  # 11
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
        calibs[channel] = {'intrin': intrin_scaled, 'extrin': T_sensor2ego}
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
    _, calib_list, ego_pose_mat = load_sample_data(sample_token,target_size=(H,W))
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
        d = np.linspace(depth_range[0]+0.5, depth_range[1]-0.5, depth_bins, dtype=np.float32)
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
        points_ego = np.einsum('ik,dhwk->dhwi', R, points_cam) + t  # (D, Hf, Wf, 3)
        # 只保留地面附近点 (z=0)
        # 但我们要将所有点投影到 BEV 网格，并记录网格索引
        x_ego = points_ego[..., 0]
        y_ego = points_ego[..., 1]
        # 计算网格索引
        gx_raw = ((x_ego - bev_range[0]) / bev_res).astype(np.int32)
        gy_raw = ((y_ego - bev_range[0]) / bev_res).astype(np.int32)
        #import pdb;pdb.set_trace()
        #gy = bev_size[0] - 1 - gx_raw   # 即使 gx_raw 是 -100，这里算出来会是 199+100=299
        gy = gx_raw   # 即使 gx_raw 是 -100，这里算出来会是 199+100=299
        gx = bev_size[1] - 1 - gy_raw
        #gx = gy_raw
        # 过滤超出范围的点
        mask = (gx >= 0) & (gx < bev_size[0]) & (gy >= 0) & (gy < bev_size[1])
        # 将有效点映射到一维索引
        grid_idx = gx * bev_size[1] + gy
        grid_idx[~mask] = -1
        #grid_idx = grid_idx[mask]
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
def generate_heatmap_gt(anns, ego_pose_mat, bev_size, class_mapping, instance_to_category, cat_name_map, sigma=3.0):
    num_classes = len(class_mapping)
    heatmap_gt = np.zeros((1, num_classes, bev_size[0], bev_size[1]), dtype=np.float32)

    def gaussian_2d(shape, center, sigma):
        y, x = np.ogrid[:shape[0], :shape[1]]
        x0, y0 = center
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    ego_pose_inv = np.linalg.inv(ego_pose_mat)
    R_global2ego = ego_pose_inv[:3, :3]
    t_global2ego = ego_pose_inv[:3, 3]

    for ann in anns:
        inst_token = ann.get('instance_token')
        if inst_token is None:
            continue
        cat_token = instance_to_category.get(inst_token)
        if cat_token is None:
            continue
        cat_name = cat_name_map.get(cat_token, 'unknown')
        class_idx = class_mapping.get(cat_name)
        if class_idx is None:
            continue  # 忽略未映射的类别

        pos_global = np.array(ann['translation'])
        pos_ego = R_global2ego @ pos_global + t_global2ego
        x, y = pos_ego[0], pos_ego[1]  # 交换（与BEV映射对齐）

        if x < BEV_RANGE[0] or x >= BEV_RANGE[1] or y < BEV_RANGE[0] or y >= BEV_RANGE[1]:
            continue

        gx = int((x - BEV_RANGE[0]) / BEV_RESOLUTION)
        gy = int((y - BEV_RANGE[0]) / BEV_RESOLUTION)
        #gx = bev_size[0] - 1 - gx
        #gy = bev_size[1] - 1 - gy

        if 0 <= gx < bev_size[0] and 0 <= gy < bev_size[1]:
            heat = gaussian_2d((bev_size[0], bev_size[1]), (gx, gy), sigma)
            heatmap_gt[0, class_idx] += heat

    heatmap_gt = np.clip(heatmap_gt, 0, 1)
    return heatmap_gt
def generate_heatmap_gt_(anns, ego_pose_mat, bev_size, sigma=3.0):
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
        x, y = pos_ego[0], pos_ego[1]
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
        x, y = pos_ego[0], pos_ego[1]
        if x < BEV_RANGE[0] or x >= BEV_RANGE[1] or y < BEV_RANGE[0] or y >= BEV_RANGE[1]:
            continue
        gx = int((x - BEV_RANGE[0]) / BEV_RESOLUTION)
        gy = int((y - BEV_RANGE[0]) / BEV_RESOLUTION)
        #gx = bev_size[0] - 1 - gx
        #gy = bev_size[1] - 1 - gy
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
import numpy as np

def min_pooling_depth_map(u, v, depth, img_h, img_w, invalid_val=-1):
    """
    生成稀疏深度图：每个像素只保留投影到该点的所有雷达点中深度最小的那个。
    
    参数：
        u, v: 像素坐标 (N,)，int 类型
        depth: 对应点的深度值 (N,)，float 类型
        img_h, img_w: 图像高度和宽度
        invalid_val: 无效像素填充值（通常用 -1）
    
    返回：
        depth_map: (img_h, img_w) 的稀疏深度图
    """
    # 1. 初始化深度图为很大的正数（这样第一次赋值时就会被替换）
    depth_map = np.full((img_h, img_w), np.inf, dtype=np.float32)
    
    # 2. 使用 np.minimum.at 进行最小池化
    # np.minimum.at(depth_map, (v, u), depth)
    # 将 depth_map[v[i], u[i]] 与 depth[i] 比较，保留较小值
    np.minimum.at(depth_map, (v, u), depth)
    
    # 3. 将未赋值的位置（仍为 np.inf）替换为 invalid_val
    depth_map[np.isinf(depth_map)] = invalid_val
    
    return depth_map
def generate_depth_gt_(sample_token, Hf, Wf):
    # 暂时使用随机深度
    depth_gt = np.random.randint(0, DEPTH_BINS, size=(1, Hf, Wf)).astype(np.int32)
    return depth_gt
def generate_depth_gt(sample_token, data_root, Hf, Wf, depth_bins=41, depth_range=(4.0, 45.0)):
    """
    从激光雷达点云生成深度真值图，仅前摄像头
    """
    # 加载 JSON
    sample_data_list = load_json('v1.0-mini/sample_data.json')
    sensors = load_json('v1.0-mini/sensor.json')
    calib_sensors = load_json('v1.0-mini/calibrated_sensor.json')
    ego_poses = load_json('v1.0-mini/ego_pose.json')
    def build_index(data_list, key='token'):
        """根据 token 建立字典索引"""
        return {item[key]: item for item in data_list}
    # 建立索引
    sample_data_idx = build_index(sample_data_list)
    calibrated_sensor_idx = build_index(calib_sensors)
    ego_pose_idx = build_index(ego_poses)
    sensor_idx = build_index(sensors)

    depth_gt_N = np.zeros((1,6,Hf,Wf),dtype=np.int32)
    idx = 0
    for CAM_CHANNEL in CAMERAS:
        # 查找 LIDAR_TOP 和摄像头
        lidar_sd = None
        cam_sd = None
        for sd in sample_data_list:
            if sd['sample_token'] != sample_token or not sd['is_key_frame']:
                continue
            calib_token = sd['calibrated_sensor_token']
            calib = calibrated_sensor_idx.get(calib_token)
            if calib is None:
                continue
            sensor = sensor_idx.get(calib.get('sensor_token'))
            if sensor is None:
                continue
            channel = sensor.get('channel')
            if channel == 'LIDAR_TOP':
                lidar_sd = sd
            elif channel == CAM_CHANNEL:
                cam_sd = sd
        if lidar_sd is None or cam_sd is None:
            return np.full((1, Hf, Wf), -1, dtype=np.int32)
    
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
        cam_rot, cam_trans, cam_intrinsic = get_calibration(cam_sd['calibrated_sensor_token'])
        # 雷达标定
        lidar_rot, lidar_trans, _ = get_calibration(lidar_sd['calibrated_sensor_token'])
        # 相机 ego_pose
        cam_ego_rot, cam_ego_trans = get_ego_pose(cam_sd['ego_pose_token'])
        # 雷达 ego_pose
        lidar_ego_rot, lidar_ego_trans = get_ego_pose(lidar_sd['ego_pose_token'])
        # 加载点云
        lidar_path = os.path.join(data_root, lidar_sd['filename'])
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3].T
    
        # ========== 5. 投影函数（四元数转旋转矩阵） ==========
        def quat_to_rot(q):
            w, x, y, z = q[0], q[1], q[2], q[3]
            return np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
                [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
            ])
    
        # 1. 雷达 -> 自车 (雷达时刻)
        R_lidar_ego = quat_to_rot(lidar_rot)
        pts = R_lidar_ego @ points + np.array(lidar_trans).reshape(3,1)
    
        # 2. 自车 -> 全局 (雷达时刻)
        R_ego_global = quat_to_rot(lidar_ego_rot)
        pts = R_ego_global @ pts + np.array(lidar_ego_trans).reshape(3,1)
    
        # 3. 全局 -> 自车 (相机时刻)
        R_global_ego = quat_to_rot(cam_ego_rot).T# 逆旋转
        pts = R_global_ego @ (pts - np.array(cam_ego_trans).reshape(3,1))
    
        # 4. 自车 -> 相机 (相机坐标系)
        R_ego_cam = quat_to_rot(cam_rot).T
        pts = R_ego_cam @ (pts - np.array(cam_trans).reshape(3,1))
    
        depths = pts[2, :]  # 深度
    
        # 5. 投影到图像
        img_pts = np.array(cam_intrinsic) @ pts  # (3, N)
        u = img_pts[0, :] / img_pts[2, :]
        v = img_pts[1, :] / img_pts[2, :]
        
        # 6. 过滤
        img_h, img_w = 900, 1600
        mask = (depths > 0) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        mask &= np.isfinite(u) & np.isfinite(v) & np.isfinite(depths)
    
        u = u[mask].astype(np.int32)
        v = v[mask].astype(np.int32)
        depth = depths[mask]
        if len(depth) == 0:
            return np.full((1, Hf, Wf), -1, dtype=np.int32)
    
        # 离散化深度
        d_min, d_max = depth_range
        depth_bins_edges = np.linspace(d_min, d_max, depth_bins+1)
        depth_indices = np.clip(np.digitize(depth, depth_bins_edges) - 1, 0, depth_bins-1)
    
        # 创建深度图
        depth_map = np.full((img_h, img_w), -1, dtype=np.int32)
        # 更稳妥：按深度降序排列，这样近处点（小深度）排在最后，赋值时覆盖前面远处的点
        order = np.argsort(depth)[::-1]  # 降序（远 -> 近）
        u_sorted = u[order]
        v_sorted = v[order]
        depth_idx_sorted = depth_indices[order]
        depth_map[v_sorted, u_sorted] = depth_idx_sorted
        depth_map[v, u] = depth_indices
        # 下采样到特征图尺寸（最近邻）
        from scipy.ndimage import zoom
        # 计算缩放因子
        scale_h = Hf / img_h
        scale_w = Wf / img_w
        depth_gt = zoom(depth_map.astype(np.float32), (scale_h, scale_w), order=0, mode='nearest')
        depth_gt = depth_gt.astype(np.int32)
        depth_gt = depth_gt[None, :, :]  # (1, Hf, Wf)
        depth_gt_N[0,idx:idx+1] = depth_gt
        idx=idx+1
    return depth_gt_N

    #depth_gt = min_pooling_depth_map(u,v,depth_map,Hf,Wf)
#return depth_gt
def count_categories(sample_tokens):
    """
    统计所有样本中出现的类别及其数量
    返回: dict {category_name: count}
    """
    from collections import defaultdict
    category_counts = defaultdict(int)
    class_counts = np.zeros(NUM_CLASSES)
    
    # 加载必要的 JSON
    sample_anns = load_json('v1.0-mini/sample_annotation.json')
    instances = load_json('v1.0-mini/instance.json')
    categories = load_json('v1.0-mini/category.json')
    inst_to_cat = {inst['token']: inst['category_token'] for inst in instances}
    cat_to_name = {cat['token']: cat['name'] for cat in categories}
    
    for ann in sample_anns:
        if ann['sample_token'] in sample_tokens:
            inst_token = ann['instance_token']
            if inst_token in inst_to_cat:
                cat_token = inst_to_cat[inst_token]
                cat_name = cat_to_name.get(cat_token, 'unknown')
                category_counts[cat_name] += 1
                idx = class_mapping.get(cat_name)
                if idx is not None:
                    class_counts[idx] += 1
    
    return category_counts,class_counts
def clip_grad_norm(grads, max_norm=1.0):
    """递归裁剪嵌套梯度的 L2 范数"""
    total_norm = 0.0
    def accumulate(g):
        nonlocal total_norm
        if isinstance(g, dict):
            for v in g.values():
                accumulate(v)
        elif isinstance(g, list):
            for v in g:
                accumulate(v)
        else:
            if g is not None:
                total_norm += np.sum(g ** 2)
    accumulate(grads)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        def scale_grads(g):
            if isinstance(g, dict):
                for k in g:
                    scale_grads(g[k])
            elif isinstance(g, list):
                for v in g:
                    scale_grads(v)
            else:
                if g is not None:
                    g *= scale
        scale_grads(grads)
    return grads
def show_pred(heatmap,reg,depth_logits_list):
    # 在训练循环中，bev_forward 返回后
    print(f"heatmap max: {heatmap.max():.4f}, min: {heatmap.min():.4f}")
    print(f"reg max: {reg.max():.4f}, min: {reg.min():.4f}")
    if depth_logits_list:
        depth_max = max([d.max() for d in depth_logits_list])
        depth_min = min([d.min() for d in depth_logits_list])
        print(f"depth_logits max: {depth_max:.4f}, min: {depth_min:.4f}")

def get_sample_from_st(sample_token,geom_indices,instance_to_category,cat_name_map,H,W,Hf,Wf):

   # 1. 加载图像和标定
    img_list, calib_list, ego_pose_mat = load_sample_data(sample_token,target_size=(H,W))
    # 转换为 (1, 6, 3, H, W) 格式
    images = np.stack([img.transpose(2,0,1) for img in img_list if img is not None], axis=0)  # (6,3,H,W)
    # 添加 batch 维度
    images = images[None, ...]  # (1,6,3,H,W)
    # 如果某个相机缺失，用零填充（但 sample 中应都有）
    # 调整顺序与 CAMERAS 一致
     # 略

    # 4. 获取标注
    anns = get_annotations(sample_token)
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

    # 5. 生成 GT
    #heatmap_gt = generate_heatmap_gt(anns, ego_pose_mat, BEV_SIZE, sigma=3.0)
    heatmap_gt = generate_heatmap_gt(anns, ego_pose_mat, BEV_SIZE, class_mapping, instance_to_category, cat_name_map, sigma=3.0)
    # 将heatmap_gt 显示为3D
    #generate_inline_html(heatmap_gt,output_html = 'view_heatmap_inline.html')
    reg_gt = generate_reg_gt(anns, ego_pose_mat, BEV_SIZE)
#depth_gt = generate_depth_gt(sample_token, Hf, Wf)
    depth_gt=generate_depth_gt(sample_token, DATA_ROOT, Hf, Wf, depth_bins=41, depth_range=(4.0, 45.0))
    sample = {} 
    sample['images'] = images
    sample['geom_indices']=geom_indices
    sample['hm']=heatmap_gt
    sample['reg']=reg_gt
    sample['depth']=depth_gt
    return sample

def get_samples(sts,geom_indices_cache,instance_to_category,H,W,Hf,Wf):
    # 加载所有必要的 JSON
    instances = load_json('v1.0-mini/instance.json')
    categories = load_json('v1.0-mini/category.json')
    instance_to_category = {inst['token']: inst['category_token'] for inst in instances}
    cat_name_map = {cat['token']: cat['name'] for cat in categories}
    sms=[]
    for st in tqdm(sts,desc="get gt and image list"):
        geom_indices = geom_indices_cache[st]
        sm=get_sample_from_st(st,geom_indices_cache[st],instance_to_category,cat_name_map,H,W,Hf,Wf)
        sms.append(sm)
    return sms

# ============================
# 训练主循环
# ============================
def main():
    BEST_MODEL_PATH = 'best_model.npz'   # 保存在当前目录
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
    sample_tokens = [s['token'] for s in samples[:3]]  # 使用前20个样本测试
    # 在 main() 中，获取样本列表后
    category_counts,class_counts = count_categories(sample_tokens)
    print("=== 类别统计 ===")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print("================")

    # 计算 alpha（与频率成反比）
    max_count = class_counts.max()
    alphas = 0.8*(1+3.0/(class_counts + 1e-6) - 3.0/(max_count+1e-6))
    #alphas = alphas / np.sum(alphas) * NUM_CLASSES  # 归一化使均值为1
    alphas = np.clip(alphas,0.01,0.99)
    print(f"Per-class alphas: {alphas}")

    # 加载所有必要的 JSON
    instances = load_json('v1.0-mini/instance.json')
    categories = load_json('v1.0-mini/category.json')
    instance_to_category = {inst['token']: inst['category_token'] for inst in instances}
    cat_name_map = {cat['token']: cat['name'] for cat in categories}



    # 确定特征图尺寸（由模型决定）
    H, W = 256, 704  # 输入图像尺寸（需与模型一致）
    #H, W = 512, 1408  # 输入图像尺寸（需与模型一致）
    Hf, Wf = H // 32, W // 32  # 8, 22

    # 预计算几何索引（耗时，可缓存）
    geom_indices_cache = {}
    for token in tqdm(sample_tokens, desc="Building geometry indices"):
        geom_indices_cache[token] = [build_geometry_indices(token, model_params, H, W, Hf, Wf)]
    samples = get_samples(sample_tokens,geom_indices_cache,instance_to_category,H,W,Hf,Wf)
    # 超参数
    lr_init = 3e-4
    epochs = 1000
    hm_weight = 0.1
    reg_weight =100.0
    depth_weight =10.0 
    # ---------- 尝试加载最佳模型 ----------
    best_loss = float('inf')
    start_epoch = 0
    m, v = None, None
    step = 1

    if os.path.exists(BEST_MODEL_PATH):
        data = np.load(BEST_MODEL_PATH, allow_pickle=True)
        model_params = data['model_params'].item()          # 恢复模型参数
        m = data['m'].item() if 'm' in data else {}        # 恢复动量
        v = data['v'].item() if 'v' in data else {}
        step = int(data['step']) if 'step' in data else 1
        best_loss = float(data['best_loss'])                # 历史最佳损失
        start_epoch = int(data['epoch']) + 1                # 从下一轮开始
        print(f"✅ 加载最佳模型 (epoch {int(data['epoch'])})，损失 {best_loss:.6f}")
    else:
        print("🆕 未找到已有模型，从头训练")
        m, v = {}, {}   # 确保后续 adam_update 能正确初始化
        step = 1
    start_epoch = 0
    best_loss = float('inf')
    for epoch in range(start_epoch,epochs):
        total_loss = 0.0
        lr = cosine_annealing(epoch,epochs,lr_init=lr_init,lr_min=1e-9) 
        for sample in tqdm([samples[2]], desc=f"Epoch {epoch+1}/{epochs}"):
            images=sample['images']
            geom_indices = sample['geom_indices']
            # 3. 前向
            heatmap, reg, depth_logits_list, bev_feat, caches = bev_forward(
                images, geom_indices, BEV_SIZE, model_params
            )
            show_pred(heatmap,reg,depth_logits_list)
            # 5. 生成 GT
            heatmap_gt = sample['hm']
			# 将heatmap_gt 显示为3D
            #generate_inline_html(sigmoid(heatmap),output_html = 'view_heatmap_infer.html')
            #save_bev_heatmap(heatmap, heatmap_gt, epoch, save_dir='bev_vis')
            save_all_class_heatmaps(heatmap, heatmap_gt, epoch, save_dir='bev_vis', num_cols=4)
            reg_gt = sample['reg']
            depth_gt=sample['depth']

            # 6. 堆叠深度 logits（取第一个相机，仅前摄像头）
            #depth_logits_batch = depth_logits_list[0][None, ...]  # (1, D, Hf, Wf)
            depth_logits_batch = np.stack(depth_logits_list,axis=0)[None,...]

            # 7. 损失计算（加权）
 
            losses, loss_caches = compute_losses(
                heatmap, reg, heatmap_gt, reg_gt,
                depth_logits_batch, depth_gt,
                hm_weight=hm_weight, reg_weight=reg_weight, depth_weight=depth_weight,
                hm_alpha = alphas,hm_gamma=2.0
            )
            total_loss += losses['total_loss']

            # 8. 梯度
            dheatmap, dreg, ddepth = d_compute_losses(
                heatmap, reg, heatmap_gt, reg_gt,
                depth_logits_batch, depth_gt, loss_caches,hm_weight=hm_weight,reg_weight = reg_weight,depth_weight=depth_weight,
                hm_alpha = alphas,hm_gamma=2.0
            )

            # 9. 构建深度梯度列表（每个相机）
            ddepth_per_cam = [None] * 6
            ddepth_per_cam = [ddepth[0, i, ...] for i in range(6)]
            # 10. 反向传播
            grads = bev_backward(
                dheatmap, dreg, ddepth_per_cam,
                caches, model_params, geom_indices, BEV_SIZE
            )
            grads = clip_grad_norm(grads,max_norm=1.0)
			# 假设 total_loss 是当前 epoch 的总损失（您代码中是单样本的 total_loss）
            if total_loss < best_loss:
                best_loss = total_loss
                np.savez_compressed(
                    BEST_MODEL_PATH,
                    model_params=model_params,
                    m=m,
                    v=v,
                    step=step,
                    epoch=epoch,
                    lr = lr,
                    best_loss=best_loss
                )
                print(f"⭐ 保存最佳模型 (epoch {epoch})，损失 {best_loss:.6f}")
            # 11. 更新
            model_params, m, v, step = adam_update(model_params, grads, lr, step, m, v)
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch:3d},lr : {lr} total: {losses['total_loss']:.6f},hm:{losses['loss_heatmap']:.6f},reg:{losses['loss_reg']:.6f},depth:{losses['loss_depth']:.6f}")

#        avg_loss = total_loss / len(sample_tokens)
#        print(f"Epoch {epoch+1}/{epochs}, avg loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
