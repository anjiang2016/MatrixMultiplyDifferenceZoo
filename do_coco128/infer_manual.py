"""
infer_manual.py
手动浏览训练集预测结果，左右方向键切换图片
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

def update_image(idx, dataset, weights, fig, axes, best_epoch, best_loss, img_size=416, conf_thresh=0.4):
    """更新图形显示指定索引的图片"""
    sample = dataset[idx]
    img = sample['image'][np.newaxis, ...]
    img_display = (sample['image'].transpose(1, 2, 0) * 255).astype(np.uint8)

    # 真实框
    true_boxes = sample['boxes']
    true_cls = sample['classes']
    true_boxes_pixel = []
    for i in range(len(true_boxes)):
        cx, cy, w, h = true_boxes[i]
        x1 = (cx - w/2) * img_size
        y1 = (cy - h/2) * img_size
        x2 = (cx + w/2) * img_size
        y2 = (cy + h/2) * img_size
        cls = int(true_cls[i])
        true_boxes_pixel.append((x1, y1, x2, y2, 1.0, cls))

    # 预测
    pred, _ = forward_yolo(img, weights)
    dets = decode_predictions(pred, img_size=img_size, conf_thresh=conf_thresh)
    pred_boxes = dets[0]

    # 置信度统计
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

    # 清空子图
    for ax in axes:
        ax.clear()
        ax.axis('off')

    # 左图：真实框
    ax = axes[0]
    ax.imshow(img_display)
    ax.set_title("Ground Truth (Green)", fontsize=10)
    for box in true_boxes_pixel:
        x1, y1, x2, y2, _, cls = box
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

    # 总体标题
    fig.suptitle(f"Image {idx+1}/{len(dataset)}  |  Best Model epoch {best_epoch}, loss {best_loss:.4f}  |  obj_max={conf_stats['obj_max']:.3f}, cls_max={conf_stats['cls_max']:.3f}", fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()

def main():
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'
    IMG_SIZE = 416
    CONF_THRESH = 0.4694

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

    # 加载数据集
    dataset, _ = load_yolo_dataset(DATA_DIR, IMG_SIZE)
    total = len(dataset)
    print(f"训练集共有 {total} 张图片，按左右方向键切换，按 q 退出")

    # 创建图形窗口
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.85)
    plt.ion()  # 交互模式，但我们会用阻塞等待

    current_idx = 0

    def on_key(event):
        nonlocal current_idx
        if event.key == 'right':
            current_idx = (current_idx + 1) % total
            update_image(current_idx, dataset, weights, fig, axes, best_epoch, best_loss, IMG_SIZE, CONF_THRESH)
        elif event.key == 'left':
            current_idx = (current_idx - 1) % total
            update_image(current_idx, dataset, weights, fig, axes, best_epoch, best_loss, IMG_SIZE, CONF_THRESH)
        elif event.key == 'q':
            plt.close(fig)

    # 连接键盘事件
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 显示第一张图片
    update_image(0, dataset, weights, fig, axes, best_epoch, best_loss, IMG_SIZE, CONF_THRESH)
    plt.show(block=True)  # 阻塞直到窗口关闭

if __name__ == "__main__":
    main()
