"""
train_yolo.py - 训练 YOLO 检测器（纯 NumPy）
"""

import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from yolo import (
    init_yolo_weights, forward_yolo, backward_yolo,
    yolo_loss, d_yolo_loss,
    decode_predictions, visualize_detections,
    load_yolo_dataset
)


def train_yolo(data_dir, img_size=416, epochs=1, batch_size=2, lr=0.001):
    """训练 YOLO 检测器（纯 NumPy）"""
    dataset, num_classes = load_yolo_dataset(data_dir, img_size)
    weights = init_yolo_weights(num_classes)

    N = len(dataset)
    num_batches = int(np.ceil(N / batch_size))
    print(f"数据集: {N} 张图片, {num_classes} 个类别")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        total_loss = 0.0
        start_time = time.time()

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, N)
            batch_indices = indices[start:end]

            batch_images = np.stack([dataset[i]['image'] for i in batch_indices], axis=0)
            batch_targets = [{'boxes': dataset[i]['boxes'], 'classes': dataset[i]['classes']}
                             for i in batch_indices]

            # 前向
            pred, caches = forward_yolo(batch_images, weights)

            # 损失
            loss = yolo_loss(pred, batch_targets, num_classes)
            total_loss += loss

            # 反向
            dloss = d_yolo_loss(pred, batch_targets, num_classes)
            grads = backward_yolo(dloss, caches)

            # 更新权重
			# 更新权重
            for key in weights:
                # 梯度裁剪（防止爆炸）
                grads[key] = np.clip(grads[key], -1.0, 1.0)
                weights[key] -= lr * grads[key]
                # 权重裁剪（防止权重过大）
                weights[key] = np.clip(weights[key], -5.0, 5.0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")

        avg_loss = total_loss / num_batches
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
        print("-" * 50)

    return weights, num_classes


def main():
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'
    IMG_SIZE = 416
    EPOCHS = 2       # 纯 NumPy 训练较慢，仅演示
    BATCH_SIZE = 2
    LR = 0.0001

    # 训练
    weights, num_classes = train_yolo(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    )

    # 保存权重
    np.savez('yolo_weights.npz', **weights)
    print("权重已保存到 yolo_weights.npz")

    # 测试单张图片
    dataset, _ = load_yolo_dataset(DATA_DIR, IMG_SIZE)
    sample = dataset[0]
    img = sample['image'][np.newaxis, ...]
    pred, _ = forward_yolo(img, weights)
    dets = decode_predictions(pred, img_size=IMG_SIZE, conf_thresh=0.3)
    img_display = (sample['image'].transpose(1, 2, 0) * 255).astype(np.uint8)

    # 显示
    visualize_detections(img_display, dets[0], title='YOLO Detection')


if __name__ == "__main__":
    main()
