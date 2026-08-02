import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ========== 1. 配置 ==========
DATAROOT = Path("/Users/zhaomingming/data_sets/v1.0-mini")
OUTPUT_DIR = Path("./map_output1")
OUTPUT_DIR.mkdir(exist_ok=True)
SAVE_INTERVAL = 1  # 每隔50帧保存一次中间结果

# ========== 2. 标定参数（从 get_bev_slow.py 复用） ==========
K = np.array([
    [1266.417203046554, 0.0,                816.2670197447984],
    [0.0,                1266.417203046554,  491.50706579294757],
    [0.0,                0.0,                1.0]
])

t = np.array([[1.70079119], [0.01594563], [1.51095764]])

quat = [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755]

def quat_to_rot(w, x, y, z):
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

R_cam_to_ego = quat_to_rot(*quat)
R_ego_to_cam = np.linalg.inv(R_cam_to_ego)
Rt_full = np.hstack([R_ego_to_cam, -R_ego_to_cam @ t])

# ========== 3. BEV 网格参数 ==========
X_MIN, X_MAX = 0.0, 20.0
Y_MIN, Y_MAX = -10.0, 10.0
RES = 0.1
Z_FIXED = 0.0

def build_lut(z_fixed=Z_FIXED):
    x_coords = np.arange(X_MIN, X_MAX, RES)
    y_coords = np.arange(Y_MIN, Y_MAX, RES)
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    X = xv.flatten()
    Y = yv.flatten()
    Z = np.full_like(X, z_fixed)
    ones = np.ones_like(X)
    
    P_ego = np.vstack([X, Y, Z, ones])
    P_cam = Rt_full @ P_ego
    Zc = P_cam[2, :]
    
    pixel_raw = K @ P_cam
    u = pixel_raw[0, :] / Zc
    v = pixel_raw[1, :] / Zc
    
    grid_shape = (len(y_coords), len(x_coords))
    return u, v, grid_shape, X, Y, Z

def ego_pose_to_transform(ego_pose):
    """从 ego_pose 构建 4×4 变换矩阵（自车 → 全局）"""
    q = ego_pose['rotation']  # [w, x, y, z]
    R = quat_to_rot(q[0], q[1], q[2], q[3])
    t = np.array(ego_pose['translation'])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# ========== 4. 主程序 ==========
print("📦 预计算 BEV 查找表...")
lut = build_lut(z_fixed=0.0)
u, v, grid_shape, X, Y, Z = lut
print(f"✅ 查找表构建完成，BEV 网格点数: {len(u)}")

# 加载 sample_data 和 ego_pose 表
sample_data_file = DATAROOT / "v1.0-mini" / "sample_data.json"
with open(sample_data_file, 'r') as f:
    sample_data = json.load(f)

ego_pose_file = DATAROOT / "v1.0-mini" / "ego_pose.json"
with open(ego_pose_file, 'r') as f:
    ego_pose_raw = json.load(f)

# 关键：将列表转为字典
if isinstance(ego_pose_raw, list):
    ego_pose_table = {item['token']: item for item in ego_pose_raw}  # 注意赋值给 ego_pose_table
else:
    ego_pose_table = ego_pose_raw

# 调试：确认类型
print("ego_pose_table 类型:", type(ego_pose_table))
print("元素个数:", len(ego_pose_table))
# 筛选 CAM_FRONT 关键帧
if isinstance(sample_data, list):
    items = sample_data
else:
    items = list(sample_data.values())

# ========== 加载数据表 ==========
# ========== 工具函数 ==========
def load_json_as_dict(filepath, key_field='token'):
    """加载 JSON 文件并转换为以 key_field 为键的字典"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return {item[key_field]: item for item in data}
    return data
sample_table = load_json_as_dict(DATAROOT / "v1.0-mini" / "sample.json")
scene_table = load_json_as_dict(DATAROOT / "v1.0-mini" / "scene.json")
log_table = load_json_as_dict(DATAROOT / "v1.0-mini" / "log.json")

# ========== 构建 sample_token → location 映射 ==========
# 1. 先构建 scene_token → location 映射
scene_location_map = {}
for scene_token, scene in scene_table.items():
    log_token = scene.get('log_token')
    if log_token and log_token in log_table:
        location = log_table[log_token].get('location')
        if location:
            scene_location_map[scene_token] = location

# 2. 构建 sample_token → location 映射
sample_location_map = {}
for sample_token, sample in sample_table.items():
    scene_token = sample.get('scene_token')
    if scene_token in scene_location_map:
        sample_location_map[sample_token] = scene_location_map[scene_token]

# ========== 城市过滤设置 ==========
# 只处理这些城市的图片（允许多个）
#ALLOWED_CITIES = ['singapore-onenorth']  # 可根据需要修改，例如 ['boston-seaport'] 或 ['singapore-onenorth', 'singapore-queenstown']
CITIES = ['singapore-onenorth','boston-seaport','singapore-onenorth', 'singapore-queenstown']
ALLOWED_CITIES = [CITIES[1]]
# 创建该城市的输出文件夹
city_dir = OUTPUT_DIR / ALLOWED_CITIES[0]
city_dir.mkdir(parents=True, exist_ok=True)

cam_front_items = []
for entry in items:
    filename = entry.get('filename', '')
    if 'samples/CAM_FRONT/' in filename and filename.endswith('.jpg'):
        sample_token = entry.get('sample_token')
        # 通过 sample_token 获取 location
        location = sample_location_map.get(sample_token)
        if location and location in ALLOWED_CITIES:
            cam_front_items.append(entry)
        else:
            # 可选：打印跳过的条目（用于调试）
            # print(f"跳过 {filename}，location={location}")
            pass

cam_front_items.sort(key=lambda x: x.get('timestamp', 0))
print(f"✅ 找到 {len(cam_front_items)} 张 CAM_FRONT 关键帧")

if len(cam_front_items) == 0:
    print("❌ 未找到 CAM_FRONT 图像，请检查数据集路径")
    exit()
# ========== 动态地图变量 ==========
map_data = np.zeros((2000, 2000, 3), dtype=np.uint8)  # 初始地图
map_offset_x = 1000  # 原点在数组中的列偏移
map_offset_y = 1000  # 原点在数组中的行偏移
map_size = 2000

def ensure_map_capacity(x, y):
    """确保地图能容纳 (x, y) 位置，必要时扩展"""
    global map_data, map_offset_x, map_offset_y, map_size
    
    # 计算在数组中的索引
    ix = x + map_offset_x
    iy = y + map_offset_y
    
    # 如果当前尺寸足够，直接返回
    if 0 <= ix < map_size and 0 <= iy < map_size:
        return
    
    # 计算需要容纳的最大坐标范围
    max_abs = max(abs(x), abs(y), map_size // 2)
    new_size = 1
    while new_size < max_abs * 2 + 1:
        new_size *= 2
    
    # 创建新地图（居中放置旧地图）
    new_map = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    dx = (new_size - map_size) // 2
    dy = (new_size - map_size) // 2
    new_map[dy:dy+map_size, dx:dx+map_size] = map_data
    
    # 更新全局变量
    map_data = new_map
    map_offset_x += dx
    map_offset_y += dy
    map_size = new_size

def set_pixel(x, y, color):
    """在地图 (x, y) 位置设置颜色"""
    ensure_map_capacity(x, y)
    ix = x + map_offset_x
    iy = y + map_offset_y
    map_data[iy, ix] = color

def get_cropped():
    """裁剪并返回有值的区域"""
    mask = np.any(map_data != 0, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return np.zeros((1, 1, 3), dtype=np.uint8)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return map_data[r_min:r_max+1, c_min:c_max+1]



# ========== 6. 遍历所有帧，累加到全局地图 ==========
print("🔄 开始拼图...")
base_x = None
base_y = None

for idx, entry in enumerate(tqdm(cam_front_items)):
    # 读取图像
    image_path = DATAROOT / entry['filename']
    if not image_path.exists():
        continue
    
    img = np.array(Image.open(image_path))
    
    # 获取 ego_pose
    ego_pose_token = entry.get('ego_pose_token')
    if ego_pose_token not in ego_pose_table:
        continue
    ego_pose = ego_pose_table[ego_pose_token]
    T_ego_to_world = ego_pose_to_transform(ego_pose)
    
    # 把 BEV 网格点转成全局坐标
    P_ego = np.vstack([X, Y, Z, np.ones_like(X)])
    P_world = T_ego_to_world @ P_ego
    X_world = P_world[0, :]
    Y_world = P_world[1, :]
    #初始化地图原点（第一帧有效数据）
    if base_x is None:
        # 使用有效点的平均位置作为原点，避免极端值
        valid_mask = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
        if np.any(valid_mask):
            base_x = np.mean(X_world[valid_mask])
            base_y = np.mean(Y_world[valid_mask])
        else:
            continue
    
    # 相对坐标
    X_rel = X_world - base_x
    Y_rel = Y_world - base_y 
    # 过滤有效像素（在图像范围内）
    mask = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
    if not np.any(mask):
        continue
    
    # 取颜色（双线性插值简化版：直接取最近邻，可优化）
    # 这里为了速度，用最近邻；如果需要更高质量，可以改用双线性插值
    u_int = np.round(u[mask]).astype(np.int64)
    v_int = np.round(v[mask]).astype(np.int64)
    u_int = np.clip(u_int, 0, img.shape[1] - 1)
    v_int = np.clip(v_int, 0, img.shape[0] - 1)
    colors = img[v_int, u_int].astype(np.float64)  # (N, 3)
    # 地图坐标（相对坐标，单位米，需要转换为像素）
    # 因为地图分辨率为 RES（米/像素），所以像素索引 = 坐标 / RES
    x_pix = (X_rel[mask] / RES).astype(np.int64)
    y_pix = (Y_rel[mask] / RES).astype(np.int64) 
    for i in range(len(x_pix)):
        set_pixel(x_pix[i],y_pix[i],colors[i])    

	# 在 for entry in tqdm(...) 循环内部，累加完成后添加：
    if idx % SAVE_INTERVAL == 0: 
        cropped = get_cropped()
        # 保存中间结果
        intermediate_path = city_dir/f"map_step_{idx:06d}.jpg"
        Image.fromarray(cropped).save(intermediate_path)
        print(f"💾 已保存中间地图: {intermediate_path}")

# ========== 7. 归一化并保存 ==========
print("🔄 归一化并保存地图...")
# 避免除以 0
count_mask = map_count > 0
map_avg = np.zeros_like(map_accum)
map_avg[count_mask] = map_accum[count_mask] / map_count[count_mask][:, None]
map_uint8 = np.clip(map_avg, 0, 255).astype(np.uint8)

# 保存
OUTPUT_MAP = city_dir /"map_final.jpg"
Image.fromarray(map_uint8).save(OUTPUT_MAP)
print(f"✅ 全局地图已保存: {OUTPUT_MAP}")
print(f"📐 地图尺寸: {map_uint8.shape[1]} × {map_uint8.shape[0]} 像素")
