"""
infer_on_training_sequential.py
逐张显示训练集图片的预测结果
左图：真实框（绿色），右图：预测框（红色）
按 Enter 下一张，输入 q 退出
"""

import sys
import os
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


def show_image_with_boxes(img_display, true_boxes_pixel, pred_boxes, idx, total, best_epoch, best_loss):
    """
    显示两张并排图片：左列真实框（绿色），右列预测框（红色）
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 左图：真实框
    ax = axes[0]
    ax.imshow(img_display)
    ax.axis('off')
    ax.set_title("Ground Truth (Green)", fontsize=12)
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
    ax.axis('off')
    ax.set_title(f"Prediction (Red)  {len(pred_boxes)} boxes", fontsize=12)
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

    fig.suptitle(f"Image {idx+1}/{total}  |  Best Model epoch {best_epoch}, loss {best_loss:.4f}", fontsize=14)
    fig.tight_layout()
    return fig


def main():
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'
    IMG_SIZE = 416
    CONF_THRESH = 0.4
    IOU_THRESH = 0.7

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
    dataset, num_classes = load_yolo_dataset(DATA_DIR, IMG_SIZE)
    total = len(dataset)
    print(f"训练集共有 {total} 张图片，将逐张显示（按 Enter 下一张，输入 q 退出）")

    for idx, sample in enumerate(dataset):
        img = sample['image'][np.newaxis, ...]
        img_display = (sample['image'].transpose(1, 2, 0) * 255).astype(np.uint8)

        # 真实框（转换为像素坐标）
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
        dets = decode_predictions(pred, img_size=IMG_SIZE, conf_thresh=CONF_THRESH,iou_thresh = IOU_THRESH)
        pred_boxes = dets[0]

        # 显示
        fig = show_image_with_boxes(img_display, true_boxes_pixel, pred_boxes,
                                    idx, total, best_epoch, best_loss)
        fig.canvas.manager.set_window_title(f"Training Image {idx+1}/{total}")

        plt.show(block=True)  # 阻塞显示
        plt.close(fig)

        # 询问是否继续
        user_input = input("按 Enter 继续下一张，输入 q 退出: ")
        if user_input.lower() == 'q':
            print("退出。")
            break

    print("所有图片显示完毕。")


if __name__ == "__main__":
    main()
