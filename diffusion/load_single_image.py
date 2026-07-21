import os
import glob
import numpy as np
from PIL import Image   # 仅用于读取图片，可换为 cv2

# ------------------------------------------------------------
# 1. 双线性插值缩放（纯 NumPy）
# ------------------------------------------------------------
def bilinear_resize(img, out_shape):
    """
    img: (H, W) float32，范围任意
    out_shape: (out_h, out_w)
    返回: (out_h, out_w) float32，范围与输入相同
    """
    in_h, in_w = img.shape
    out_h, out_w = out_shape

    scale_h = in_h / out_h
    scale_w = in_w / out_w

    out_y = np.arange(out_h, dtype=np.float32) * scale_h + 0.5 * (scale_h - 1)
    out_x = np.arange(out_w, dtype=np.float32) * scale_w + 0.5 * (scale_w - 1)

    y0 = np.floor(out_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x0 = np.floor(out_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, in_w - 1)

    dy = out_y - y0
    dx = out_x - x0

    y0 = y0[:, None]
    y1 = y1[:, None]
    dy = dy[:, None]

    I00 = img[y0, x0]
    I01 = img[y0, x1]
    I10 = img[y1, x0]
    I11 = img[y1, x1]

    top = I00 + (I01 - I00) * dx
    bottom = I10 + (I11 - I10) * dx
    result = top + (bottom - top) * dy
    return result

# ------------------------------------------------------------
# 2. RGB转灰度（纯 NumPy）
# ------------------------------------------------------------
def rgb_to_grayscale(rgb_img):
    """
    rgb_img: (H, W, 3) float32 或 uint8，范围 [0,255] 或 [0,1]
    返回: (H, W) float32，范围保持与输入一致（即若输入 [0,255]，输出 [0,255]）
    """
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("输入必须是 (H, W, 3) 的 RGB 图像")
    # 统一转为 float32 并确定最大值
    if rgb_img.dtype == np.uint8:
        rgb_img = rgb_img.astype(np.float32)
        max_val = 255.0
    else:
        max_val = 1.0 if rgb_img.max() <= 1.0 else 255.0
    # 标准灰度权重
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.dot(rgb_img[..., :3], weights)   # (H, W)
    # 保证数值范围不变（若输入为0-255，输出也为0-255）
    return gray

# ------------------------------------------------------------
# 3. 加载单张图片（从文件或文件夹）
# ------------------------------------------------------------
def load_single_image(path, target_size=(32, 32)):
    """
    加载一张彩色图片，转为灰度，缩放到 target_size，归一化到 [-1,1]。
    - path: 图片文件路径或文件夹路径（随机选一张）
    - target_size: (H, W)
    返回: (1, 1, H, W) float32，范围 [-1,1]
    """
    # 处理文件夹
    if os.path.isdir(path):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(path, e)))
        if not files:
            raise ValueError(f"在文件夹 {path} 中未找到图片")
        img_path = np.random.choice(files)
        print(f"随机选择图片: {img_path}")
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        img_path = path

    # 读取图片（使用 PIL，可替换为 cv2.imread）
    img = Image.open(img_path).convert('RGB')   # 确保 RGB
    rgb = np.array(img, dtype=np.uint8)         # (H, W, 3) uint8
    import pdb;pdb.set_trace()
    # 转灰度
    gray = rgb_to_grayscale(rgb)                # (H, W) float32, 范围 0-255

    # 缩放到目标尺寸
    gray_resized = bilinear_resize(gray, target_size)   # (H, W) float32

    # 归一化到 [-1,1]
    gray_norm = (gray_resized / 127.5) - 1.0

    # 添加 batch 和 channel 维度
    gray_norm = gray_norm.reshape(1, 1, target_size[0], target_size[1])
    return gray_norm

# ------------------------------------------------------------
# 可选：调试保存（纯 NumPy + PIL 保存）
# ------------------------------------------------------------
def save_debug_image(arr, save_path='debug.png'):
    """保存 (1,1,H,W) 到 PNG，范围 [-1,1] 自动映射到 0-255"""
    img = ((arr[0, 0] + 1.0) * 127.5).astype(np.uint8)
    Image.fromarray(img, mode='L').save(save_path)
    print(f"调试图像已保存: {save_path}")



x0 = load_single_image('./imgs')
save_debug_image(x0)
