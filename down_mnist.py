"""
download_mnist.py
下载 MindSpore 版本的 MNIST，解压并转换为 pkl
自动识别 train 和 test 文件夹下的 IDX 文件
"""

import os
import pickle
import numpy as np
import requests
import zipfile
import glob
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================
URL = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
ZIP_NAME = "MNIST_Data.zip"
EXTRACT_DIR = "MNIST_Data"
OUTPUT_PKL = "mnist.pkl"


# ============================================================
# 下载
# ============================================================
def download_file(url, filename):
    print(f"正在下载: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print("下载完成！")


# ============================================================
# 读取 IDX 格式
# ============================================================
def read_idx_image(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)


def read_idx_label(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


# ============================================================
# 查找文件
# ============================================================
def find_file(directory, pattern):
    """在目录下查找匹配 pattern 的文件，返回完整路径"""
    files = glob.glob(os.path.join(directory, pattern))
    if len(files) == 0:
        # 尝试递归查找
        files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"在 {directory} 下找不到匹配 {pattern} 的文件")
    return files[0]


# ============================================================
# 主流程
# ============================================================
def main():
    # ===== 检查是否已存在 =====
    if os.path.exists(OUTPUT_PKL):
        print(f"✅ {OUTPUT_PKL} 已存在，跳过下载和转换。")
        return
    # 1. 下载
    if not os.path.exists(ZIP_NAME):
        download_file(URL, ZIP_NAME)
    else:
        print(f"{ZIP_NAME} 已存在，跳过下载")

    # 2. 解压（强制解压覆盖）
    print(f"正在解压 {ZIP_NAME} ...")
    with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("解压完成！")

    # 3. 查找文件（自动适应命名）
    train_dir = os.path.join(EXTRACT_DIR, "train")
    test_dir = os.path.join(EXTRACT_DIR, "test")

    # 训练集图像和标签
    train_img = find_file(train_dir, "*images*")
    train_lbl = find_file(train_dir, "*labels*")

    # 测试集图像和标签
    test_img = find_file(test_dir, "*images*")
    test_lbl = find_file(test_dir, "*labels*")

    print(f"训练图像: {os.path.basename(train_img)}")
    print(f"训练标签: {os.path.basename(train_lbl)}")
    print(f"测试图像: {os.path.basename(test_img)}")
    print(f"测试标签: {os.path.basename(test_lbl)}")

    # 4. 读取数据
    print("读取训练集...")
    X_train = read_idx_image(train_img)
    y_train = read_idx_label(train_lbl)

    print("读取测试集...")
    X_test = read_idx_image(test_img)
    y_test = read_idx_label(test_lbl)

    # 5. 拆分 55000/5000
    X_train, X_val = X_train[:55000], X_train[55000:]
    y_train, y_val = y_train[:55000], y_train[55000:]

    # 归一化到 [0,1]
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # 转为 (N, 1, 28, 28)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_val = X_val.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    data = ((X_train, y_train), (X_val, y_val), (X_test, y_test))

    # 6. 保存 pkl
    print(f"保存到 {OUTPUT_PKL} ...")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ 转换完成！")
    print(f"  训练集: {X_train.shape[0]} 张")
    print(f"  验证集: {X_val.shape[0]} 张")
    print(f"  测试集: {X_test.shape[0]} 张")


if __name__ == "__main__":
    main()
