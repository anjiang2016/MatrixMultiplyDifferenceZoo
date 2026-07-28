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
BEV_RANGE = (-50, 50)          # 米
BEV_RESOLUTION = 0.1           # 米/像素
BEV_SIZE = int((BEV_RANGE[1] - BEV_RANGE[0]) / BEV_RESOLUTION)

# ========== 工具函数 ==========
def load_json(rel_path):
    with open(os.path.join(DATA_ROOT, rel_path), 'r') as f:
        return json.load(f)
def get_category_color(cat_name):
    if cat_name == 'vehicle.car':
        return 'blue'
    elif cat_name == 'vehicle.truck':
        return 'orange'
    elif cat_name == 'vehicle.bus':
        return 'purple'
    elif cat_name == 'vehicle.bicycle':
        return 'green'
    elif cat_name == 'vehicle.motorcycle':
        return 'lime'
    else:
        return 'red'  # 其他车辆
def load_lane_lines_from_map(map_file_path):
    """
    从 nuScenes-map-expansion-v1.3 地图 JSON 文件中提取车道线点序列。
    返回 list of (x, y) points (全局坐标，米制)
    """
    with open(map_file_path, 'r') as f:
        data = json.load(f)
    
    # 1. 构建 node token -> (x, y) 映射
    node_coords = {}
    for node in data.get('node', []):
        token = node.get('token')
        x = node.get('x')
        y = node.get('y')
        if token is not None and x is not None and y is not None:
            node_coords[token] = (x, y)
    
    # 2. 提取 line 要素（所有 line，或者根据 type 过滤）
    lines = data.get('line', [])
    lane_points = []
    
    for line in lines:
        # 如果有 type，可过滤，否则全部提取
        # line_type = line.get('type')
        # if line_type not in ['road_divider', 'lane']: continue
        
        node_tokens = line.get('node_tokens', [])
        if len(node_tokens) < 2:
            continue
        
        points = []
        for token in node_tokens:
            if token in node_coords:
                points.append(node_coords[token])
        if len(points) >= 2:
            lane_points.append(points)
    
    return lane_points
def draw_lane_lines(ax, lane_points_global, ego_pose_mat, bev_range, bev_res, bev_size,
                    color='white', linewidth=3):
    """
    在 BEV 图上绘制车道线。
    参数:
        ax: matplotlib axes
        lane_points_global: 从地图中提取的车道线点列表（全局坐标）
        ego_pose_mat: (4,4) 自车→全局变换矩阵
        bev_range, bev_res, bev_size: BEV 参数
        color, linewidth: 绘制样式
    """
    R_global2ego = ego_pose_mat[:3, :3].T
    t_global2ego = -R_global2ego @ ego_pose_mat[:3, 3]
#    R_global2ego = ego_pose_mat[:3, :3]
#    t_global2ego = ego_pose_mat[:3, 3]
    def ego_to_pixel(x, y):
        gx = int((x - bev_range[0]) / bev_res)
        gy = int((y - bev_range[0]) / bev_res)
        gx = bev_size - 1 - gx
        gy = bev_size - 1 - gy
        if 0 <= gx < bev_size and 0 <= gy < bev_size:
            return gx, gy
        return None, None

    for points in lane_points_global:
        ego_points = []
        for (x_global, y_global) in points:
            # 全局坐标转自车坐标（忽略 z）
            pos_global = np.array([x_global, y_global, 0])
            pos_ego = R_global2ego @ pos_global + t_global2ego
            x_ego, y_ego = pos_ego[1], pos_ego[0]
            px, py = ego_to_pixel(x_ego, y_ego)
            if px is not None:
                ego_points.append((px, py))
        # 绘制折线（如果有多个点）
        if len(ego_points) >= 2:
            xs, ys = zip(*ego_points)
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.7)
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
        # 传感器->自车
        R_ego2sensor = quaternion_to_rotation_matrix(calib_token_to_sensor[calib_token]['rotation'])
        t_ego2sensor = np.array(calib_token_to_sensor[calib_token]['translation'])
        T_sensor2ego = np.eye(4)
        T_sensor2ego[:3, :3] = R_ego2sensor
        T_sensor2ego[:3, 3] = t_ego2sensor
        calibs[channel] = {'intrin': intrin, 'extrin': T_sensor2ego}
    img_list = [images.get(cam) for cam in ORDER]
    calib_list = [calibs.get(cam) for cam in ORDER]
    return img_list, calib_list

def project_camera_to_bev(img, intrin, extrin, bev_range, bev_res, step=2, flip_y=True):
    """
    将单张相机图像投影到 BEV 网格
    参数:
        img: (H, W, 3) 图像
        intrin: (3,3) 内参矩阵
        extrin: (4,4) 相机→自车变换矩阵
        bev_range: (min, max)
        bev_res: 分辨率
        step: 采样步长
        flip_y: 是否翻转 y 轴使左侧在左
    返回:
        grid_x, grid_y, colors
    """
    # 图像归一化
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

    H, W, _ = img.shape
    fx, fy = intrin[0,0], intrin[1,1]
    cx, cy = intrin[0,2], intrin[1,2]

    uu, vv = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    u = uu.ravel()
    v = vv.ravel()

    x_c = (u - cx) / fx
    y_c = (v - cy) / fy
    rays = np.stack([x_c, y_c, np.ones_like(u)], axis=1)

    R_cam2ego = extrin[:3, :3]
    t_cam2ego = extrin[:3, 3]

    d_ego = (R_cam2ego @ rays.T).T
    valid = d_ego[:, 2] < 0
    if not np.any(valid):
        return np.array([]), np.array([]), np.array([])

    d_ego = d_ego[valid]
    rays = rays[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    lam = -t_cam2ego[2] / d_ego[:, 2]
    P_ego = lam[:, None] * d_ego + t_cam2ego

    x_ego = P_ego[:, 1]
    y_ego = P_ego[:, 0]

    bev_size = int((bev_range[1] - bev_range[0]) / bev_res)
    idx = (x_ego >= bev_range[0]) & (x_ego < bev_range[1]) & (y_ego >= bev_range[0]) & (y_ego < bev_range[1])
    if not np.any(idx):
        return np.array([]), np.array([]), np.array([])

    x_ego = x_ego[idx]
    y_ego = y_ego[idx]
    u_valid = u_valid[idx]
    v_valid = v_valid[idx]

    grid_x = ((x_ego - bev_range[0]) / bev_res).astype(np.int32)
    grid_y = ((y_ego - bev_range[0]) / bev_res).astype(np.int32)

#    grid_x = bev_size - 1 - grid_x   # 翻转 x 使前方在上
#    if flip_y:
#        grid_y = bev_size - 1 - grid_y

    grid_x = np.clip(grid_x, 0, bev_size - 1)
    grid_y = np.clip(grid_y, 0, bev_size - 1)

    colors = img[v_valid, u_valid]
    return grid_x, grid_y, colors

# ========== 标注处理 ==========
def transform_points(points, rot, trans):
    """将点从全局转换到自车（或反之）"""
    return (rot @ points.T).T + trans

def get_annotation_category(ann, instance_to_category, cat_name_map):
    inst_token = ann['instance_token']
    if inst_token in instance_to_category:
        cat_token = instance_to_category[inst_token]
        return cat_name_map.get(cat_token, 'unknown')
    return 'unknown'
def draw_ego_vehicle(ax, bev_range, bev_res, bev_size,
                     car_length=5.5*3, car_width=2.1*3,
                     color='white', alpha=0.3, edgecolor='cyan'):
    """
    在 BEV 图像上绘制自车矩形，并标注 'Ego'。
    参数:
        ax: matplotlib axes
        bev_range: (min, max) BEV 范围（米）
        bev_res: 分辨率
        bev_size: 网格尺寸
        car_length: 自车长度（米）
        car_width: 自车宽度（米）
        color: 填充颜色
        alpha: 透明度
        edgecolor: 边框颜色
    """
    def ego_to_pixel(x, y):
        gx = int((x - bev_range[0]) / bev_res)
        gy = int((y - bev_range[0]) / bev_res)
#        gx = bev_size - 1 - gx
#        gy = bev_size - 1 - gy
        if 0 <= gx < bev_size and 0 <= gy < bev_size:
            return gx, gy
        return None, None

    ego_x, ego_y = 0, 0
    half_l = car_length / 2.0
    half_w = car_width / 2.0
    # 交换 x 和 y 方向，使长度沿 y，宽度沿 x，与标注框对齐
    corners = np.array([
        [-half_w, -half_l*0.4],
        [ half_w, -half_l*0.4],
        [ half_w,  half_l*1.6],
        [-half_w,  half_l*1.6]
    ])
    # 自车朝向为 x 轴正向，在 BEV 中前方在上
    rot = np.eye(2)
    corners_rot = (rot @ corners.T).T
    corners_ego = corners_rot + np.array([ego_x, ego_y])

    pixels = []
    for cx, cy in corners_ego:
        px, py = ego_to_pixel(cx, cy)
        if px is not None:
            pixels.append((px, py))

    if len(pixels) == 4:
        polygon = plt.Polygon(pixels, closed=True, fill=True,
                              facecolor=color, alpha=alpha,
                              edgecolor=edgecolor, linewidth=2)
        ax.add_patch(polygon)
        # 中心标注
        cx_pix, cy_pix = ego_to_pixel(0, 2)
        if cx_pix is not None:
            ax.text(cx_pix, cy_pix, 'Ego', color=edgecolor, fontsize=8,
                    weight='bold', ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.4,
                              edgecolor='none', pad=1))

def draw_bev_annotations(ax, sample_token, sample_anns, instance_to_category, cat_name_map,
                         ego_pose_mat, bev_range, bev_res, bev_size):
    """
    在 BEV 图像上绘制标注框和类别
    参数:
        ax: matplotlib axes
        sample_token: 当前样本 token
        sample_anns: 所有标注列表
        instance_to_category: 映射
        cat_name_map: 类别名称映射
        ego_pose_mat: (4,4) 自车→全局
        bev_range: (min, max)
        bev_res: 分辨率
        bev_size: 网格尺寸
    """
    draw_ego_vehicle(ax, bev_range, bev_res, bev_size,
                     car_length=5.5, car_width=2.0)
    anns = [ann for ann in sample_anns if ann['sample_token'] == sample_token]
    if not anns:
        return

    T_global2ego = np.linalg.inv(ego_pose_mat)
#T_global2ego = ego_pose_mat

    def ego_to_pixel(x, y):
        gx = int((x - bev_range[0]) / bev_res)
        gy = int((y - bev_range[0]) / bev_res)
#        gx = bev_size - 1 - gx
#        gy = bev_size - 1 - gy
        if 0 <= gx < bev_size and 0 <= gy < bev_size:
            return gx, gy
        return None, None

    for ann in anns:
        # ---- 只显示车辆类别（以 'vehicle.' 开头） ----
        cat_name = get_annotation_category(ann, instance_to_category, cat_name_map)
        if not cat_name.startswith('vehicle.'):
            continue
        # 全局坐标 → 自车坐标
        pos_global = np.array(ann['translation'])
        pos_ego = transform_points(pos_global, T_global2ego[:3, :3], T_global2ego[:3, 3])
        x, y = pos_ego[0], pos_ego[1]

        # 尺寸和朝向
        w, l, h = ann['size']
        quat = ann['rotation']  # 全局四元数
        # 计算全局偏航角
        yaw_global = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                                1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
        # 转换方向向量到自车
        dir_global = np.array([np.cos(yaw_global), np.sin(yaw_global), 0])
        dir_ego = (T_global2ego[:3, :3] @ dir_global)[:2]
        yaw_ego = np.arctan2(dir_ego[1], dir_ego[0])

        half_l = l / 2.0
        half_w = w / 2.0
        corners = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ])
        rot = np.array([
            [np.cos(yaw_ego), -np.sin(yaw_ego)],
            [np.sin(yaw_ego),  np.cos(yaw_ego)]
        ])
        corners_rot = (rot @ corners.T).T
        corners_ego = corners_rot + np.array([x, y])

        pixels = []
        for cx, cy in corners_ego:
            px, py = ego_to_pixel(cx, cy)
            if px is not None:
                pixels.append((px, py))

        if len(pixels) == 4:
            polygon = plt.Polygon(pixels, closed=True, fill=None, edgecolor='red', linewidth=2)
            ax.add_patch(polygon)

            # 类别标签
            cx_pix, cy_pix = ego_to_pixel(x, y)
            if cx_pix is not None:
                cat_name = get_annotation_category(ann, instance_to_category, cat_name_map)
                color = get_category_color(cat_name)
                ax.text(cx_pix, cy_pix-10, cat_name,
                        color=color, fontsize=8, weight='bold', ha='center',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                polygon = plt.Polygon(pixels, closed=True, fill=True, facecolor=color,edgecolor=color, linewidth=2)
                ax.add_patch(polygon)

def get_lane_points_for_sample(sample_token):
    # 查找 sample
    sample = None
    for s in samples:
        if s['token'] == sample_token:
            sample = s
            break
    if sample is None:
        return []
    scene_token = sample.get('scene_token')
    if scene_token is None:
        return []
    log_token = scene_token_to_log_token.get(scene_token)
    if log_token is None:
        return []
    location = log_token_to_location.get(log_token)
    if location is None:
        return []
    map_file = os.path.join(DATA_ROOT, f"maps/expansion/{location}.json")
    if map_file not in map_cache:
        if os.path.exists(map_file):
            map_cache[map_file] = load_lane_lines_from_map(map_file)
            print(f"加载 {location} 地图，车道线 {len(map_cache[map_file])} 条")
        else:
            map_cache[map_file] = []
            print(f"地图文件不存在: {map_file}")
    return map_cache[map_file]
map_element_cache = {}  # 缓存地图元素字典

def get_map_elements_for_sample(sample_token):
    """返回完整地图元素字典"""
    # 查找 sample -> scene -> log -> location 的代码与 get_lane_points_for_sample 相同
    # 这里复用 location 查找逻辑，可提取为单独函数
    scene_token = None
    for s in samples:
        if s['token'] == sample_token:
            scene_token = s.get('scene_token')
            break
    if scene_token is None:
        return None
    log_token = scene_token_to_log_token.get(scene_token)
    if log_token is None:
        return None
    location = log_token_to_location.get(log_token)
    if location is None:
        return None

    map_file = os.path.join(DATA_ROOT, f"maps/expansion/{location}.json")
    if map_file not in map_element_cache:
        if os.path.exists(map_file):
            map_element_cache[map_file] = load_map_elements(map_file)
            print(f"加载 {location} 地图元素: "
                  f"车道线 {len(map_element_cache[map_file]['lane'])} 条, "
                  f"道路边界 {len(map_element_cache[map_file]['road_divider'])} 条, "
                  f"停车线 {len(map_element_cache[map_file]['stop_line'])} 条, "
                  f"人行横道 {len(map_element_cache[map_file]['crosswalk'])} 个")
        else:
            map_element_cache[map_file] = None
            print(f"地图文件不存在: {map_file}")
    return map_element_cache.get(map_file)
def load_map_elements(map_file_path):
    with open(map_file_path, 'r') as f:
        data = json.load(f)

    # 构建 node token -> (x, y) 映射
    node_coords = {}
    for node in data.get('node', []):
        token = node.get('token')
        x = node.get('x')
        y = node.get('y')
        if token is not None and x is not None and y is not None:
            node_coords[token] = (x, y)

    # 构建 line token -> 点列表 映射
    line_coords = {}
    for line in data.get('line', []):
        token = line.get('token')
        node_tokens = line.get('node_tokens', [])
        points = []
        for nt in node_tokens:
            if nt in node_coords:
                points.append(node_coords[nt])
        if len(points) >= 2:
            line_coords[token] = points

    # 构建 polygon token -> 顶点列表 映射
    polygon_coords = {}
    for poly in data.get('polygon', []):
        token = poly.get('token')
        node_tokens = poly.get('exterior_node_tokens', [])
        points = []
        for nt in node_tokens:
            if nt in node_coords:
                points.append(node_coords[nt])
        if len(points) >= 3:
            polygon_coords[token] = points

    elements = {
        'lane': [],          # 车道线（从 line 提取，但未被 road_divider 引用）
        'road_divider': [],  # 道路分隔线（从 road_divider 引用）
        'crosswalk': [],     # 人行横道（从 ped_crossing 引用）
        'polygons': []       # 其他多边形
    }

    # 1. 获取被 road_divider 引用的 line_token
    divider_line_tokens = set()
    for rd in data.get('road_divider', []):
        line_token = rd.get('line_token')
        if line_token in line_coords:
            elements['road_divider'].append(line_coords[line_token])
            divider_line_tokens.add(line_token)

    # 2. 提取未被 road_divider 引用的 line 作为车道线
    for token, points in line_coords.items():
        if token not in divider_line_tokens:
            elements['lane'].append(points)

    # 3. 提取人行横道（ped_crossing → polygon）
    for pc in data.get('ped_crossing', []):
        poly_token = pc.get('polygon_token')
        if poly_token in polygon_coords:
            elements['crosswalk'].append(polygon_coords[poly_token])

    # 4. 其他多边形（未被 ped_crossing 引用的）
    used_polygons = set(pc.get('polygon_token') for pc in data.get('ped_crossing', []))
    for token, points in polygon_coords.items():
        if token not in used_polygons:
            elements['polygons'].append(points)

    return elements
def draw_map_elements(ax, map_elements, ego_pose_mat, bev_range, bev_res, bev_size,
                      color_lane='yellow', color_divider='white', 
                      color_crosswalk='gray', alpha_crosswalk=0.4,
                      color_polygon=(0.2, 1.0, 0.8), alpha_polygon=0.3):
    R_global2ego = ego_pose_mat[:3, :3].T
    t_global2ego = -R_global2ego @ ego_pose_mat[:3, 3]

    def ego_to_pixel(x, y):
        gx = int((x - bev_range[0]) / bev_res)
        gy = int((y - bev_range[0]) / bev_res)
#        gx = bev_size - 1 - gx
#        gy = bev_size - 1 - gy
        if 0 <= gx < bev_size and 0 <= gy < bev_size:
            return gx, gy
        return None, None

    def transform_points(points_global):
        pixels = []
        for (x_global, y_global) in points_global:
            pos_global = np.array([x_global, y_global, 0])
            pos_ego = R_global2ego @ pos_global + t_global2ego
            x_ego, y_ego = pos_ego[0], pos_ego[1]
            px, py = ego_to_pixel(x_ego, y_ego)
            if px is not None:
                pixels.append((px, py))
        return pixels

    # 车道线（黄色实线）
    for points in map_elements.get('lane', []):
        pixels = transform_points(points)
        if len(pixels) >= 2:
            xs, ys = zip(*pixels)
            ax.plot(xs, ys, color=color_lane, linewidth=1.5, alpha=0.8)

    # 道路分隔线（白色虚线）
    for points in map_elements.get('road_divider', []):
        pixels = transform_points(points)
        if len(pixels) >= 2:
            xs, ys = zip(*pixels)
            ax.plot(xs, ys, color=color_divider, linewidth=1, linestyle='--', alpha=0.7)

    # 人行横道（灰色半透明填充）
    for points in map_elements.get('crosswalk', []):
        pixels = transform_points(points)
        if len(pixels) >= 3:
            polygon = plt.Polygon(pixels, closed=True, fill=True,
                                  facecolor=color_crosswalk, alpha=alpha_crosswalk,
                                  edgecolor='white', linewidth=1)
            ax.add_patch(polygon)

    # 其他多边形（浅灰色半透明填充）
    for points in map_elements.get('polygons', []):
        pixels = transform_points(points)
        if len(pixels) >= 3:
            polygon = plt.Polygon(pixels, closed=True, fill=True,
                                  facecolor=color_polygon, alpha=alpha_polygon,
                                  edgecolor='none')
            ax.add_patch(polygon)
# ========== 主程序 ==========
if __name__ == "__main__":
    # 加载所有必要的 JSON
    print("加载数据...")
    samples = load_json('v1.0-mini/sample.json')
    sample_tokens = [s['token'] for s in samples]
    total = len(sample_tokens)
    # 在 __main__ 开头加载数据后，建立映射
    samples = load_json('v1.0-mini/sample.json')
    scenes = load_json('v1.0-mini/scene.json')
    logs = load_json('v1.0-mini/log.json')

    scene_token_to_log_token = {scene['token']: scene['log_token'] for scene in scenes}
    log_token_to_location = {log['token']: log['location'] for log in logs}

    map_cache = {}
    # 加载标注
    sample_anns = load_json('v1.0-mini/sample_annotation.json')
    instances = load_json('v1.0-mini/instance.json')
    categories = load_json('v1.0-mini/category.json')
    instance_to_category = {inst['token']: inst['category_token'] for inst in instances}
    cat_name_map = {cat['token']: cat['name'] for cat in categories}
    ego_poses = load_json('v1.0-mini/ego_pose.json')
    ego_pose_map = {ep['token']: ep for ep in ego_poses}

    # 创建输出目录
    output_dir = "bev_output_with_annotations"
    os.makedirs(output_dir, exist_ok=True)

    print(f"共 {total} 个样本，开始生成带标注的 BEV 序列...")

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

            # 获取当前样本的 ego_pose（从第一个关键帧取）
            sample_data_list = load_json('v1.0-mini/sample_data.json')
            ego_pose_token = None
            for sd in sample_data_list:
                if sd['sample_token'] == sample_token and sd['is_key_frame']:
                    ego_pose_token = sd['ego_pose_token']
                    break
            if ego_pose_token is None:
                print(f"  警告：无法获取 ego_pose_token，跳过标注")
                # 直接保存无标注图像
                plt.imsave(os.path.join(output_dir, f"bev_{idx+1:04d}_{sample_token[:8]}.png"), bev)
            else:
                ego_pose = ego_pose_map[ego_pose_token]
                ego_pose_mat = np.eye(4)
                ego_pose_mat[:3, :3] = quaternion_to_rotation_matrix(ego_pose['rotation'])
                ego_pose_mat[:3, 3] = ego_pose['translation']

                # 创建图形并绘制标注
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(bev)
                # 在绘制 BEV 的循环中，在 ax.imshow(bev) 之后，绘制标注之前或之后
                # 在循环内，获取 ego_pose_mat 之后
#                lane_points_global = get_lane_points_for_sample(sample_token)
#                if lane_points_global:
#                    draw_lane_lines(ax, lane_points_global, ego_pose_mat,BEV_RANGE, BEV_RESOLUTION, BEV_SIZE,color='yellow', linewidth=1)
                # 绘制地图元素
                map_elements = get_map_elements_for_sample(sample_token)
                if map_elements:
                    draw_map_elements(ax, map_elements, ego_pose_mat,BEV_RANGE, BEV_RESOLUTION, BEV_SIZE)
                draw_bev_annotations(ax, sample_token, sample_anns, instance_to_category, cat_name_map,
                                     ego_pose_mat, BEV_RANGE, BEV_RESOLUTION, BEV_SIZE)
                ax.axis('off')
                plt.tight_layout()
                save_path = os.path.join(output_dir, f"bev_{idx+1:04d}_{sample_token[:8]}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            if (idx + 1) % 10 == 0 or idx == total - 1:
                print(f"  已保存 {idx+1} 张图片")

        except Exception as e:
            print(f"  处理样本 {sample_token} 时出错: {e}")
            continue

    print("全部完成！")
