import json
import numpy as np
from pathlib import Path

# 尝试导入 pyproj（用于经纬度转换）
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# ========== 配置路径 ==========
DATAROOT = Path("/Users/zhaomingming/data_sets/v1.0-mini")  # 请修改为你的实际路径

# ========== 工具函数 ==========
def load_json_as_dict(filepath, key_field='token'):
    """加载 JSON 文件并转换为以 key_field 为键的字典"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return {item[key_field]: item for item in data}
    return data

# ========== 1. 加载所有必要的表 ==========
print("📂 加载数据表...")
sample_data_table = load_json_as_dict(DATAROOT / "v1.0-mini" / "sample_data.json")
sample_table = load_json_as_dict(DATAROOT / "v1.0-mini" / "sample.json")
scene_table = load_json_as_dict(DATAROOT / "v1.0-mini" / "scene.json")
ego_pose_table = load_json_as_dict(DATAROOT / "v1.0-mini" / "ego_pose.json")
log_table = load_json_as_dict(DATAROOT / "v1.0-mini" / "log.json")
map_table = load_json_as_dict(DATAROOT / "v1.0-mini" / "map.json")

print(f"✅ 加载完成")
print(f"   sample_data: {len(sample_data_table)} 条")
print(f"   sample: {len(sample_table)} 条")
print(f"   scene: {len(scene_table)} 条")
print(f"   ego_pose: {len(ego_pose_table)} 条")
print(f"   log: {len(log_table)} 条")
print(f"   map: {len(map_table)} 条")

# ========== 2. 选择一张 CAM_FRONT 图片 ==========
# 方式1：从 sample_data 中找一条 CAM_FRONT 的 jpg（关键帧或sweeps均可）
target_sample_data = None
for entry in sample_data_table.values():
    filename = entry.get('filename', '')
    if 'samples/CAM_FRONT/' in filename and filename.endswith('.jpg'):
        target_sample_data = entry
        break

if target_sample_data is None:
    print("❌ 未找到 CAM_FRONT 图像")
    exit()
sample_token = target_sample_data.get('sample_token')
print(f"\n📷 图片: {target_sample_data['filename']}")
print(f"   sample_token: {sample_token}")

# ========== 3. 通过 sample_token 获取 scene_token 和 ego_pose_token ==========
sample = sample_table.get(sample_token)
if sample is None:
    print(f"❌ sample_token {sample_token} 不在 sample_table 中")
    exit()

scene_token = sample.get('scene_token')
ego_pose_token = target_sample_data.get('ego_pose_token')

print(f"   scene_token: {scene_token}")
print(f"   ego_pose_token: {ego_pose_token}")

# ========== 4. 通过 scene_token 获取 log_token ==========
scene = scene_table.get(scene_token)
if scene is None:
    print(f"❌ scene_token {scene_token} 不在 scene_table 中")
    exit()

log_token = scene.get('log_token')
print(f"   log_token: {log_token}")

# ========== 5. 通过 log_token 获取 location ==========
log_entry = log_table.get(log_token)
if log_entry is None:
    print(f"❌ log_token {log_token} 不在 log_table 中")
    exit()

location = log_entry.get('location')
print(f"   location: {location}")


# ========== 7. 通过 ego_pose_token 获取车辆的平移量 ==========
ego_pose = ego_pose_table.get(ego_pose_token)
if ego_pose is None:
    print(f"❌ ego_pose_token {ego_pose_token} 不在 ego_pose_table 中")
    exit()

translation = ego_pose.get('translation')
print(f"   ego_pose translation: {translation}")

# ========== 免库硬编码偏移量 ==========
# 根据 v1.0-mini 数据反推的偏移量（仅适用于 singapore-onenorth）
# 真实 UTM = 偏移量 + translation
#UTM_OFFSET = [364963.178, 143138.140, 0.0]
UTM_OFFSET_MAP = {
    'singapore-onenorth': [364963.178, 143138.140, 0.0],  # 手动调校值
    'singapore-queenstown':[317200.0, 143000.0],
    'singapore-hollandvillage':[317100.0, 141500.0],
    'boston': [322500.0, 4678500.0, 0.0],            # 近似值
}
location='singapore-queenstown'
# 获取偏移量
utm_offset = UTM_OFFSET_MAP.get(location, [0.0, 0.0, 0.0])

print(f"📐 UTM 偏移量 (utm_offset): {utm_offset}")
# ========== 8. 计算真实 UTM 坐标 ==========
utm_x = utm_offset[0]+ translation[0]
utm_y = utm_offset[1]+ translation[1]
utm_z =  translation[2]  # 通常为 0

print(f"\n🌐 真实 UTM 坐标:")
print(f"   X (东向): {utm_x:.3f} 米")
print(f"   Y (北向): {utm_y:.3f} 米")
print(f"   Z: {utm_z:.3f} 米")

# ========== 9. 转换为经纬度（需要 pyproj） ==========
if HAS_PYPROJ:
    try:
        # 根据 location 选择 UTM 分区
        # 波士顿: UTM Zone 19N (EPSG: 32619)
        # 新加坡: UTM Zone 48N (EPSG: 32648)
        if 'boston' in location.lower():
            utm_zone = 19
            epsg = 32619
        elif 'singapore' in location.lower():
            utm_zone = 48
            epsg = 32648
        else:
            print(f"⚠️ 未知的 location: {location}，无法确定 UTM Zone")
            exit()

        # 方法1：使用 pyproj 直接转换
        utm_proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84', south=False)
        lon, lat = utm_proj(utm_x, utm_y, inverse=True)
        
        print(f"\n🌍 经纬度 (WGS84):")
        print(f"   经度 (Longitude): {lon:.6f}°")
        print(f"   纬度 (Latitude):  {lat:.6f}°")
         # 生成 URL
        apple_url = f"https://maps.apple.com.cn/search?center={lat:.6f}%2C{lon:.6f}&span=0.0214893%2C0.0209807"
        print(f"🔗 苹果地图链接: {apple_url}") 
        # 方法2：使用 EPSG 代码（更可靠）
        # from pyproj import Transformer
        # transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326")
        # lon, lat = transformer.transform(utm_x, utm_y)
        # print(f"   (EPSG) 经度: {lon:.6f}, 纬度: {lat:.6f}")
        
    except Exception as e:
        print(f"⚠️ 经纬度转换失败: {e}")
else:
    print("\n⚠️ pyproj 未安装，无法转换为经纬度")
    print("   安装方法: pip install pyproj")
