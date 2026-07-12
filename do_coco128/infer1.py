"""
infer_on_training_auto.py
自动播放训练集预测结果，每0.5秒切换一张
显示置信度统计信息
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from yolo import (
    forward_yolo, decode_predictions,
    load_yolo_dataset
)

# COCO 类别名称
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

def show_image_auto(img_display, true_boxes_pixel, pred_boxes, idx, total, best_epoch, best_loss, conf_stats, fig, axes):
    """
    更新图形内容（不创建新窗口）
    """
    # 清空子图
    for ax in axes:
        ax.clear()
        ax.axis('off')
    
    # 左图：真实框
    ax = axes[0]
    ax.imshow(img_display)
    ax.set_title("Ground Truth (Green)", fontsize=10)
    for box in true_boxes_pixel:
        x1, y1, x2, y2, conf, cls = box
        cls = int(cls)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='green', facecolor='none'
        )
        ax.add_patch(rect)
        if cls < len(COCO_CLASSES):
            label = COCO_CLASSES[cls]
        else:
            label = f"cls{cls}"
        ax.text(x1, y1 - 5, label,
                color='white', fontsize=8,
                bbox=dict(facecolor='green', alpha=0.7, edgecolor='none', pad=1))
    
    # 右图：预测框
    ax = axes[1]
    ax.imshow(img_display)
    ax.set_title(f"Prediction (Red) {len(pred_boxes)} boxes, conf max={conf_stats['conf_max']:.3f}", fontsize=10)
    for box in pred_boxes:
        x1, y1, x2, y2, conf, cls = box
        cls = int(cls)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        if cls < len(COCO_CLASSES):
            label = f"{COCO_CLASSES[cls]}: {conf:.2f}"
        else:
            label = f"cls{cls}: {conf:.2f}"
        ax.text(x1, y1 - 20, label,
                color='white', fontsize=8,
                bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=1))
    
    fig.suptitle(f"Image {idx+1}/{total}  |  Best Model epoch {best_epoch}, loss {best_loss:.4f}  |  obj_max={conf_stats['obj_max']:.3f}, cls_max={conf_stats['cls_max']:.3f}", fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.5)  # 显示0.5秒

def main():
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'
    IMG_SIZE = 416
    CONF_THRESH = 0.4

    # 加载最佳模型
    best_model_path = 'best_model.npz'
    if not os.path.exists(best_model_path):
        print(f"❌ 未找到最佳模型文件: {best_model_path}")
        return

    best_data = np.load(best_model_path, allow_pickle=True)
    weights = {k: best_data[k] for k in best_data.files if k not in ['best_loss', 'best_epoch']}
    best_loss = float(best_data.get('best_loss', float('inf')))
    best_epoch = int(best_data.get('best_epoch', 0))
    print(f"✅ 加载最佳模型 (epoch {best_epoch}, loss {best_loss:.4f})")

    # 加载训练集
    dataset, _ = load_yolo_dataset(DATA_DIR, IMG_SIZE)
    total = len(dataset)
    print(f"训练集共有 {total} 张图片，将自动播放（每0.5秒一张）")

    # 创建图形窗口
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.ion()  # 交互模式
    
    for idx, sample in enumerate(dataset):
        img = sample['image'][np.newaxis, ...]
        img_display = (sample['image'].transpose(1, 2, 0) * 255).astype(np.uint8)

        # 真实框
        true_boxes = sample['boxes']
        true_cls = sample['classes']
        true_boxes_pixel = []
        for i in range(len(true_boxes)):
            cx, cy, w, h = true_boxes[i]
            x1 = (cx - w/2) * IMG_SIZE
            y1 = (cy - h/2) * IMG_SIZE
            x2 = (cx + w/2) * IMG_SIZE
            y2 = (cy + h/2) * IMG_SIZE
            cls = int(true_cls[i])
            true_boxes_pixel.append((x1, y1, x2, y2, 1.0, cls))

        # 预测
        pred, _ = forward_yolo(img, weights)
        dets = decode_predictions(pred, img_size=IMG_SIZE, conf_thresh=CONF_THRESH)
        pred_boxes = dets[0]

        # 计算置信度统计
        pred_batch = pred[0]
        obj_logits = pred_batch[4, :, :]
        cls_logits = pred_batch[5:, :, :]
        obj = 1 / (1 + np.exp(-obj_logits))
        cls = 1 / (1 + np.exp(-cls_logits))
        cls_max = np.max(cls, axis=0)
        conf = obj * cls_max
        conf_stats = {
            'obj_max': float(obj.max()),
            'cls_max': float(cls_max.max()),
            'conf_max': float(conf.max())
        }

        # 显示
        show_image_auto(img_display, true_boxes_pixel, pred_boxes, idx, total, best_epoch, best_loss, conf_stats, fig, axes)
        
        # 每10张打印一次进度
        if (idx+1) % 10 == 0:
            print(f"已显示 {idx+1}/{total} 张")

    plt.ioff()
    plt.show(block=True)
    print("播放完毕。")

if __name__ == "__main__":
    main()
