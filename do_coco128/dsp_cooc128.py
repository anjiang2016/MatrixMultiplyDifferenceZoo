"""
show_coco128.py
循环显示 COCO128 数据集图片及标注框
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# COCO 80 类别名称（与 YOLO 格式索引对应）
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def load_coco128_dataset(data_dir, split='train2017'):
    """
    加载 YOLO 格式的 COCO128 数据集
    返回: 图片路径列表, 标签列表 (每个标签为 (类别, x_center, y_center, w, h) 归一化)
    """
    img_dir = os.path.join(data_dir, 'images', split)
    label_dir = os.path.join(data_dir, 'labels', split)
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    dataset = []
    
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append((cls_id, x_c, y_c, w, h))
        
        dataset.append((img_path, boxes))
    
    return dataset


def show_image_with_boxes(img_path, boxes, ax, class_names=COCO_CLASSES):
    """
    在指定 ax 上显示图片并绘制边界框
    """
    # 读取图片
    img = Image.open(img_path).convert('RGB')
    img_width, img_height = img.size
    
    # 显示图片
    ax.imshow(img)
    ax.axis('off')
    
    # 绘制每个框
    for cls_id, x_c, y_c, w, h in boxes:
        # 转换为像素坐标 (左上角 + 宽高)
        x1 = (x_c - w / 2) * img_width
        y1 = (y_c - h / 2) * img_height
        width = w * img_width
        height = h * img_height
        
        # 创建矩形
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加类别名称
        class_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        ax.text(x1, y1 - 5, class_name,
                color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.6, edgecolor='none'))


def main():
    # 数据路径（请根据实际情况修改）
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'  # 修改为你的路径
    #DATA_DIR = './coco128_32x32'  # 修改为你的路径
    
    # 加载数据集
    dataset = load_coco128_dataset(DATA_DIR, split='train2017')
    if not dataset:
        print(f"未在 {DATA_DIR} 找到图片，请检查路径。")
        return
    
    print(f"加载了 {len(dataset)} 张图片。")
    
    # 初始化 matplotlib 图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    current_idx = 0
    
    def update_image(idx):
        """更新显示第 idx 张图片"""
        ax.clear()
        img_path, boxes = dataset[idx]
        show_image_with_boxes(img_path, boxes, ax)
        ax.set_title(f"{idx+1}/{len(dataset)}: {os.path.basename(img_path)}")
        fig.canvas.draw()
    
    def on_key(event):
        nonlocal current_idx
        if event.key == 'right' or event.key == 'n':
            current_idx = (current_idx + 1) % len(dataset)
            update_image(current_idx)
        elif event.key == 'left' or event.key == 'p':
            current_idx = (current_idx - 1) % len(dataset)
            update_image(current_idx)
        elif event.key == 'q':
            plt.close(fig)
    
    # 绑定键盘事件
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # 显示第一张
    update_image(0)
    print("操作说明: 按 'n' 或 '→' 下一张，按 'p' 或 '←' 上一张，按 'q' 退出。")
    plt.show()


if __name__ == '__main__':
    main()
