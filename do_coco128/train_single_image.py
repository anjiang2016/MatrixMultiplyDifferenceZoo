"""
test_single_image_overfit.py
用单张图片训练模型，验证模型是否能完全过拟合
"""

import sys
import os
import numpy as np
import time
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from yolo import (
    init_yolo_weights, forward_yolo, backward_yolo,
    yolo_loss, d_yolo_loss,
    load_yolo_dataset
)

def train_single_image():
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'
    IMG_SIZE = 416
    
    # 加载数据集，只取第一张图片
    dataset, num_classes = load_yolo_dataset(DATA_DIR, IMG_SIZE)
    sample = dataset[0]  # 只用第一张
    print(f"使用第1张图片，包含 {len(sample['boxes'])} 个目标")
    
    # 初始化模型
    weights = init_yolo_weights(num_classes)
    lr = 0.001
    
    # 准备单张图片的数据
    x = sample['image'][np.newaxis, ...]  # (1, 3, 416, 416)
    targets = [{'boxes': sample['boxes'], 'classes': sample['classes']}]
    
    print("开始单样本训练，目标：完全过拟合...")
    print("-" * 50)
    
    for epoch in range(1, 501):  # 训练500个epoch
        pred, caches = forward_yolo(x, weights)
        loss, loss_comp = yolo_loss(pred, targets, num_classes)
        
        dloss = d_yolo_loss(pred, targets, num_classes)
        grads = backward_yolo(dloss, caches)
        
        # 更新权重
        for key in weights:
            weights[key] -= lr * grads[key]
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss:.6f}")
            # 打印置信度统计（objectness最大值）
            obj = 1 / (1 + np.exp(-pred[0, 4, :, :]))
            cls = 1 / (1 + np.exp(-pred[0, 5:, :, :]))
            conf = obj * np.max(cls, axis=0)
            print(f"  obj_max: {obj.max():.4f}, cls_max: {np.max(cls):.4f}, conf_max: {conf.max():.4f}")
        
        # 如果 Loss 足够低，提前结束
        if loss < 0.01:
            print(f"🎉 Epoch {epoch} 达到 Loss < 0.01，停止训练")
            break
    
    # 最终推理，看预测结果
    pred, _ = forward_yolo(x, weights)
    from yolo import decode_predictions
    dets = decode_predictions(pred, img_size=IMG_SIZE, conf_thresh=0.01)
    print(f"\n最终预测框数: {len(dets[0])}")
    for box in dets[0]:
        print(f"  {box}")
    
    return weights

if __name__ == "__main__":
    train_single_image()
