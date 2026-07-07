import os
import json
import requests
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO
from pycocotools.coco import COCO

# --- 1. 配置 ---
# 数据集下载URL（来自Ultralytics官方）[reference:0][reference:1]
DATASET_URL = "https://ultralytics.com/assets/coco128.zip"
DATASET_ZIP = "coco128.zip"
DATASET_DIR = "/Users/zhaomingming/data_sets/coco128"

# --- 2. 下载并解压数据集（如果本地没有）---
if not os.path.exists(DATASET_DIR):
    print(f"正在下载数据集: {DATASET_URL}")
    try:
        response = requests.get(DATASET_URL, stream=True)
        response.raise_for_status()
        with open(DATASET_ZIP, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("下载完成，正在解压...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(DATASET_ZIP)
        print("解压完成！")
    except Exception as e:
        print(f"下载或解压失败: {e}")
        exit()
else:
    print("数据集已存在，跳过下载。")

# --- 3. 加载COCO格式标注 ---
# 标注文件路径[reference:2]
ann_file = os.path.join(DATASET_DIR, "annotations", "instances_train2017.json")
# 图片文件夹路径[reference:3]
img_dir = os.path.join(DATASET_DIR, "images", "train2017")

# 初始化COCO API
coco = COCO(ann_file)

# 获取所有图片的ID
img_ids = coco.getImgIds()
print(f"数据集共包含 {len(img_ids)} 张图片。")

# --- 4. 显示图片及其标注框 ---
def show_images_with_bboxes(num_images=4):
    """
    随机选择并显示几张图片及其检测框
    
    Args:
        num_images: 要显示的图片数量
    """
    # 随机选择要显示的图片ID
    selected_ids = np.random.choice(img_ids, min(num_images, len(img_ids)), replace=False)
    
    # 创建子图布局
    cols = 2
    rows = (len(selected_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    # 如果只有一个子图，将axes转换为二维数组以便统一处理
    if len(selected_ids) == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_id in enumerate(selected_ids):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        # 获取图片信息
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        # 读取图片
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"图片文件未找到: {img_path}")
            continue
        
        # 显示图片
        ax.imshow(image)
        
        # 获取该图片的所有标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # 在图片上绘制边界框和标签
        for ann in anns:
            # 获取边界框坐标 (x, y, width, height)
            bbox = ann['bbox']
            x, y, w, h = bbox
            # 创建矩形框
            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=2, 
                edgecolor='red', 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # 获取类别名称
            cat_id = ann['category_id']
            cat_name = coco.loadCats(cat_id)[0]['name']
            # 添加类别标签
            ax.text(
                x, y - 5, 
                cat_name, 
                color='white', 
                fontsize=10, 
                bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', pad=1)
            )
        
        ax.set_title(f"Image ID: {img_id}")
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(selected_ids), rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# 显示4张图片
if __name__ == "__main__":
    show_images_with_bboxes(num_images=4)
