import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ========== 1. 配置 ==========
DATAROOT = Path("/Users/zhaomingming/data_sets/v1.0-mini")
OUTPUT_DIR = Path("./map_output")
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
CITIES = ['singapore-onenorth','boston-seaport','singapore-hollandvillage', 'singapore-queenstown']
ALLOWED_CITIES = [CITIES[0]]
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

# ========== 5. 初始化全局地图 ==========
# 先遍历所有帧，计算全局地图的范围（为了确定地图尺寸）
print("📐 计算全局地图范围...")
all_x_world, all_y_world = [], []

for entry in tqdm(cam_front_items, desc="扫描范围"):
    ego_pose_token = entry.get('ego_pose_token')
    if ego_pose_token not in ego_pose_table:
        continue
    ego_pose = ego_pose_table[ego_pose_token]
    T_ego_to_world = ego_pose_to_transform(ego_pose)
    
    P_ego = np.vstack([X, Y, Z, np.ones_like(X)])
    P_world = T_ego_to_world @ P_ego
    all_x_world.append(P_world[0, :])
    all_y_world.append(P_world[1, :])

all_x_world = np.concatenate(all_x_world)
all_y_world = np.concatenate(all_y_world)

x_min, x_max = all_x_world.min(), all_x_world.max()
y_min, y_max = all_y_world.min(), all_y_world.max()
print(f"📐 地图范围: X [{x_min:.1f}, {x_max:.1f}], Y [{y_min:.1f}, {y_max:.1f}]")

# 地图分辨率（米/像素），与 BEV 分辨率一致
MAP_RES = RES
map_width = int(np.ceil((x_max - x_min) / MAP_RES))
map_height = int(np.ceil((y_max - y_min) / MAP_RES))
print(f"📐 地图尺寸: {map_width} × {map_height} 像素")

# 初始化累加器（用 float 累加，最后转 uint8）
map_accum = np.zeros((map_height, map_width, 3), dtype=np.float64)
map_count = np.zeros((map_height, map_width), dtype=np.uint32)

# ========== 6. 遍历所有帧，累加到全局地图 ==========
print("🔄 开始拼图...")
def crop_map(map_img):
    """裁剪掉全黑的行和列（只保留有值区域）"""
    # 找出所有非零像素的行和列
    rows = np.any(map_img != 0, axis=(1, 2))
    cols = np.any(map_img != 0, axis=(0, 2))
    
    if not np.any(rows) or not np.any(cols):
        # 如果全黑，返回原图（或返回最小尺寸）
        return map_img
    
    return map_img[np.ix_(rows, cols)]
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
    
    # 计算全局地图上的像素位置
    col = ((X_world - x_min) / MAP_RES).astype(np.int64)
    #row = ((Y_world - y_min) / MAP_RES).astype(np.int64)
    row = ((y_max - Y_world) / MAP_RES).astype(np.int64)  # 用 y_max 减 
    # 过滤有效像素（在图像范围内）
    mask = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
    mask &= (col >= 0) & (col < map_width) & (row >= 0) & (row < map_height)
    
    if not np.any(mask):
        continue
    
    # 取颜色（双线性插值简化版：直接取最近邻，可优化）
    # 这里为了速度，用最近邻；如果需要更高质量，可以改用双线性插值
    u_int = np.round(u[mask]).astype(np.int64)
    v_int = np.round(v[mask]).astype(np.int64)
    u_int = np.clip(u_int, 0, img.shape[1] - 1)
    v_int = np.clip(v_int, 0, img.shape[0] - 1)
    colors = img[v_int, u_int].astype(np.float64)  # (N, 3)
    
    # 累加到地图
    col_valid = col[mask]
    row_valid = row[mask]
    
    #np.add.at(map_accum, (row_valid, col_valid), colors)
    #np.add.at(map_count, (row_valid, col_valid), 1)
	
    map_accum[row_valid,col_valid]=colors
    map_count[row_valid,col_valid]=1

	# 在 for entry in tqdm(...) 循环内部，累加完成后添加：
    if idx % SAVE_INTERVAL == 0: 
        # 计算当前平均地图
        count_mask = map_count > 0
        map_avg = np.zeros_like(map_accum)
        map_avg[count_mask] = map_accum[count_mask] / map_count[count_mask][:, None]
        map_uint8 = np.clip(map_avg, 0, 255).astype(np.uint8)

        # 保存中间结果
        intermediate_path = city_dir/f"map_step_{idx:06d}.jpg"
        #Image.fromarray(crop_map(map_uint8)).save(intermediate_path)
        Image.fromarray(map_uint8).save(intermediate_path)
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
