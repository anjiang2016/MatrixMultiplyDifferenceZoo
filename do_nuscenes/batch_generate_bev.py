# batch_generate_bev.py
import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 从你已有的 get_bev.py 中导入核心函数
from get_bev import build_lut, generate_bev_fast, K, Rt_full, X_MIN, X_MAX, Y_MIN, Y_MAX, RES

# ========== 1. 配置 ==========
DATAROOT = Path("/Users/zhaomingming/data_sets/v1.0-mini")
OUTPUT_DIR = Path("./bev_frames")
OUTPUT_DIR.mkdir(exist_ok=True)

# ========== 2. 读取所有 CAM_FRONT 图像元数据 ==========
sample_data_file = DATAROOT / "v1.0-mini" / "sample_data.json"
with open(sample_data_file, 'r') as f:
    sample_data = json.load(f)

# 如果是列表，直接遍历；如果是字典，取 values
if isinstance(sample_data, list):
    items = sample_data
else:
    items = list(sample_data.values())

# 筛选所有 CAM_FRONT 图像（包括 samples 和 sweeps，按需调整）
cam_front_items = []
for entry in items:
    filename = entry.get('filename', '')
    if 'samples/CAM_FRONT/' in filename and filename.endswith('.jpg'):
        cam_front_items.append(entry)

# 按时间戳排序
cam_front_items.sort(key=lambda x: x.get('timestamp', 0))
print(f"✅ 找到 {len(cam_front_items)} 张 CAM_FRONT 图像")

# ========== 3. 预计算 BEV 查找表（所有帧共享） ==========
print("📦 预计算 BEV 查找表...")
lut = build_lut(z_fixed=0.0)
print("✅ 查找表构建完成")

# ========== 4. 批量生成 BEV 帧 ==========
print("🔄 开始生成 BEV 帧...")
for idx, entry in enumerate(tqdm(cam_front_items)):
    image_path = DATAROOT / entry['filename']
    
    # 检查图片是否存在
    if not image_path.exists():
        print(f"⚠️ 图片不存在，跳过: {image_path}")
        continue
    
    try:
        # 读取图像
        img = np.array(Image.open(image_path))
        
        # 生成 BEV 图
        bev = generate_bev_fast(img, lut)
        
        # 保存为帧（用 6 位数字编号，方便 ffmpeg 合成）
        frame_name = f"frame_{idx:06d}.jpg"
        frame_path = OUTPUT_DIR / frame_name
        # 将 BEV 图缩放到与原图高度一致（保持宽高比）
        w_orig = img.shape[1]
        h_bev = int(bev.shape[0] * w_orig / bev.shape[1])
        bev_resized = np.array(Image.fromarray(bev).resize((w_orig, h_bev)))
        
        # 左右拼接
        combined = np.vstack([img, bev_resized])
        Image.fromarray(combined).save(frame_path)
        
    except Exception as e:
        print(f"❌ 处理 {image_path} 时出错: {e}")
        continue

print(f"✅ 全部完成！共生成 {len(cam_front_items)} 帧，保存在 {OUTPUT_DIR}")
