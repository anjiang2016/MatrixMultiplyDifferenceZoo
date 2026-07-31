import json
import numpy as np
from PIL import Image
from pathlib import Path

dataroot = Path("/Users/zhaomingming/data_sets/v1.0-mini") # 用户必须在此处修改

# 1. 加载 sample_data
with open(dataroot / "v1.0-mini" / "sample_data.json", 'r') as f:
    sample_data = json.load(f)

# 打印前几个字符，确认结构（调试用）
print(f"📄 sample_data 类型: {type(sample_data)}")
# 如果 sample_data 是列表，直接遍历；如果是字典，取 values()
if isinstance(sample_data, list):
    items = sample_data
elif isinstance(sample_data, dict):
    items = list(sample_data.values())
else:
    raise TypeError("sample_data 格式异常")

print(f"📄 数据集条目总数: {len(items)}")
# ========== 3. 从 filename 中提取传感器类型 ==========
def extract_sensor_from_filename(filename):
    """从 filename 中解析传感器名称，如 'samples/CAM_FRONT/xxx.jpg' -> 'CAM_FRONT'"""
    parts = filename.split('/')
    if len(parts) >= 2:
        return parts[1]  # 第二级目录，如 'CAM_FRONT'
    return None

# 扫描所有条目，建立传感器名称列表
sensor_set = set()
sensor_entries = {}  # 传感器名称 -> 第一个匹配的条目

for entry in items:
    filename = entry.get('filename', '')
    sensor = extract_sensor_from_filename(filename)
    if sensor:
        sensor_set.add(sensor)
        if sensor not in sensor_entries:
            sensor_entries[sensor] = entry

print("\n📋 数据集中找到的传感器类型:")
for s in sorted(sensor_set):
    print(f"   - {s}")

# ========== 4. 查找目标传感器（CAM_FRONT） ==========
target_sensor = "CAM_FRONT"

if target_sensor in sensor_entries:
    entry = sensor_entries[target_sensor]
    print(f"\n✅ 找到 {target_sensor} 的数据")
else:
    # 尝试不区分大小写匹配
    match = [s for s in sensor_set if s.lower() == target_sensor.lower()]
    if match:
        entry = sensor_entries[match[0]]
        print(f"\n✅ 找到近似匹配: {match[0]}")
    else:
        print(f"\n❌ 未找到 {target_sensor}，请检查数据集是否包含相机数据")
        print("可用的传感器类型:")
        for s in sorted(sensor_set):
            print(f"   - {s}")
        exit(1)


sample_token = entry.get('sample_token')
calib_token = entry.get('calibrated_sensor_token')
filename = entry.get('filename')
img_path = dataroot / filename

print(f"找到 sample_token: {sample_token}")
print(f"图像路径: {img_path}")

# 3. 加载图像
img = np.array(Image.open(img_path))

# 4. 加载标定数据
with open(dataroot / "v1.0-mini" / "calibrated_sensor.json", 'r') as f:
    calib_data = json.load(f)
# 兼容 dict/list
if isinstance(calib_data, dict):
    calib_entry = calib_data.get(calib_token)
elif isinstance(calib_data, list):
    calib_entry = next((item for item in calib_data if item.get('token') == calib_token), None)
else:
    raise TypeError("calibrated_sensor.json 格式异常")

if calib_entry is None:
    raise ValueError(f"未找到标定 token: {calib_token}")

K = np.array(calib_entry['camera_intrinsic'])
translation = np.array(calib_entry['translation'])
rotation_quat = calib_entry['rotation'] # [w, x, y, z]

# 5. 四元数到旋转矩阵
def quat_to_rot(w, x, y, z):
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])
R = quat_to_rot(*rotation_quat)

print("内参 K:\n", K)
print("平移 t:", translation)
print(f"🔄 四元数 (w,x,y,z): {rotation_quat}")
print(f"🧭 旋转矩阵 R:\n{R}")
