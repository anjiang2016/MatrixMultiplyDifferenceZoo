"""
test_single_image_overfit.py
用单张图片训练模型，验证模型是否能完全过拟合，并可视化预测结果
"""

import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from yolo import (
    init_yolo_weights, forward_yolo, backward_yolo,
    yolo_loss, d_yolo_loss,
    load_yolo_dataset, decode_predictions
)

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

def visualize_result(img_display, pred_boxes, true_boxes_pixel, title):
    """可视化真实框（绿）和预测框（红）"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_display)
    ax.axis('off')
    ax.set_title(title)
    
    # 真实框（绿色）
    for box in true_boxes_pixel:
        x1, y1, x2, y2, cls = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, COCO_CLASSES[cls], color='white', fontsize=10,
                bbox=dict(facecolor='green', alpha=0.7, edgecolor='none'))
    
    # 预测框（红色）
    for box in pred_boxes:
        x1, y1, x2, y2, conf, cls = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-25, f'{COCO_CLASSES[cls]}: {conf:.2f}', color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.7, edgecolor='none'))
    
    plt.show()

def train_single_image():
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'
    IMG_SIZE = 416
    
    dataset, num_classes = load_yolo_dataset(DATA_DIR, IMG_SIZE)
    sample = dataset[0]
    print(f"使用第1张图片，包含 {len(sample['boxes'])} 个目标")
    print(f"目标类别: {[COCO_CLASSES[int(c)] for c in sample['classes']]}")
    
    weights = init_yolo_weights(num_classes)
    lr = 0.001
    
    x = sample['image'][np.newaxis, ...]
    targets = [{'boxes': sample['boxes'], 'classes': sample['classes']}]
    
    print("开始单样本训练...")
    print("-" * 50)
    
    for epoch in range(1, 501):
        pred, caches = forward_yolo(x, weights)
        loss, loss_comp = yolo_loss(pred, targets, num_classes)
        
        dloss = d_yolo_loss(pred, targets, num_classes)
        grads = backward_yolo(dloss, caches)
        
        for key in weights:
            weights[key] -= lr * grads[key]
        
        if epoch % 50 == 0:
            obj = 1 / (1 + np.exp(-pred[0, 4, :, :]))
            cls = 1 / (1 + np.exp(-pred[0, 5:, :, :]))
            conf = obj * np.max(cls, axis=0)
            print(f"Epoch {epoch:3d}, Loss: {loss:.6f}, obj_max: {obj.max():.4f}, conf_max: {conf.max():.4f}")
        
        if loss < 0.01:
            print(f"🎉 Epoch {epoch} 达到 Loss < 0.01，停止训练")
            break
    
    # 最终推理并可视化
    pred, _ = forward_yolo(x, weights)
    dets = decode_predictions(pred, img_size=IMG_SIZE, conf_thresh=0.1)
    pred_boxes = dets[0]
    
    # 真实框（转像素坐标）
    true_boxes_pixel = []
    for i in range(len(sample['boxes'])):
        cx, cy, w, h = sample['boxes'][i]
        x1 = (cx - w/2) * IMG_SIZE
        y1 = (cy - h/2) * IMG_SIZE
        x2 = (cx + w/2) * IMG_SIZE
        y2 = (cy + h/2) * IMG_SIZE
        true_boxes_pixel.append((x1, y1, x2, y2, int(sample['classes'][i])))
    
    img_display = (sample['image'].transpose(1,2,0) * 255).astype(np.uint8)
    print(f"\n预测框数: {len(pred_boxes)}")
    for box in pred_boxes:
        print(f"  {COCO_CLASSES[int(box[5])]}: {box[4]:.3f}")
    
    visualize_result(img_display, pred_boxes, true_boxes_pixel, 
                     f"Single Image Overfit Test: {len(true_boxes_pixel)} GT, {len(pred_boxes)} Pred")

if __name__ == "__main__":
    train_single_image()
