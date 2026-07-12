"""
show_coco128_resized.py
将 COCO128 图片和标注缩放到 32x32 显示
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# COCO 80 类别名称
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
    """加载 COCO128 数据集，返回 (图片路径, 标注列表) 的列表"""
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


def resize_image_and_boxes(img_path, boxes, target_size=32):
    """
    将图片缩放到 target_size x target_size，并返回缩放后的图像和调整后的框坐标（像素）。
    框坐标：每个框为 (cls_id, x1, y1, x2, y2) 像素坐标（左上角、右下角）。
    """
    # 读取原始图片
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    
    # 缩放到 target_size
    img_resized = img.resize((target_size, target_size), Image.LANCZOS)
    img_array = np.array(img_resized)
    
    # 调整框坐标
    new_boxes = []
    for cls_id, x_c, y_c, w, h in boxes:
        # 将归一化坐标转换为缩放后图像上的像素坐标
        x1 = (x_c - w / 2) * target_size
        y1 = (y_c - h / 2) * target_size
        x2 = (x_c + w / 2) * target_size
        y2 = (y_c + h / 2) * target_size
        # 限制在图像范围内
        x1 = max(0, min(target_size, x1))
        y1 = max(0, min(target_size, y1))
        x2 = max(0, min(target_size, x2))
        y2 = max(0, min(target_size, y2))
        new_boxes.append((cls_id, x1, y1, x2, y2))
    
    return img_array, new_boxes


def show_resized_image(img_array, boxes, ax, class_names=COCO_CLASSES):
    """
    显示缩放后的图像和框
    """
    ax.imshow(img_array, interpolation='nearest')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 绘制框
    for cls_id, x1, y1, x2, y2 in boxes:
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=1, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 类别标签
        class_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        ax.text(x1, y1 - 2, class_name,
                color='white', fontsize=6,
                bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', pad=0.5))


def main():
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'  # 请修改为你的路径
    TARGET_SIZE = 32
    
    dataset = load_coco128_dataset(DATA_DIR, split='train2017')
    if not dataset:
        print(f"未在 {DATA_DIR} 找到图片，请检查路径。")
        return
    
    print(f"加载了 {len(dataset)} 张图片。")
    
    # 预加载所有缩放后的图像和框（加速浏览）
    resized_data = []
    for img_path, boxes in dataset:
        img_array, new_boxes = resize_image_and_boxes(img_path, boxes, TARGET_SIZE)
        resized_data.append((img_array, new_boxes))
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    current_idx = 0
    
    def update_image(idx):
        ax.clear()
        img_array, boxes = resized_data[idx]
        show_resized_image(img_array, boxes, ax)
        ax.set_title(f"{idx+1}/{len(dataset)}  (32x32)", fontsize=10)
        fig.canvas.draw()
    
    def on_key(event):
        nonlocal current_idx
        if event.key in ('right', 'n'):
            current_idx = (current_idx + 1) % len(dataset)
            update_image(current_idx)
        elif event.key in ('left', 'p'):
            current_idx = (current_idx - 1) % len(dataset)
            update_image(current_idx)
        elif event.key == 'q':
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update_image(0)
    print("操作说明: 按 'n' 或 '→' 下一张，按 'p' 或 '←' 上一张，按 'q' 退出。")
    plt.show()


if __name__ == '__main__':
    main()
