"""
train_yolo.py - 训练 YOLO 检测器（纯 NumPy），集成实时网页监控
"""

import sys
import os
import json
import time
import numpy as np
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

# ============================================================
# 日志记录模块
# ============================================================
LOG_FILE = "training_log.json"
def init_log():
    return {
        "epochs": [],
        "losses": [],
        "loss_components": {"box": [], "cls": [], "obj": []},
        "learning_rates": [],
        "times": [],
        "layer_weights": {},   # 添加这行
        "layer_grads": {}      # 添加这行
    }

def write_log(log_data):
    """写入 JSON 文件"""
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)

def log_metrics(log_data, epoch, loss, loss_components, avg_weight, lr, epoch_time):
    """记录一个 epoch 的指标"""
    log_data["epochs"].append(epoch)
    log_data["losses"].append(float(loss))
    log_data["loss_components"]["box"].append(float(loss_components.get("box", 0)))
    log_data["loss_components"]["cls"].append(float(loss_components.get("cls", 0)))
    log_data["loss_components"]["obj"].append(float(loss_components.get("obj", 0)))
    log_data["avg_weights"].append(float(avg_weight))
    log_data["learning_rates"].append(float(lr))
    log_data["times"].append(float(epoch_time))
    write_log(log_data)

# ============================================================
# 训练函数
# ============================================================
def train_yolo(data_dir, img_size=416, epochs=1, batch_size=2, lr=0.001):
    dataset, num_classes = load_yolo_dataset(data_dir, img_size)
    weights = init_yolo_weights(num_classes)

    N = len(dataset)
    num_batches = int(np.ceil(N / batch_size))
    print(f"数据集: {N} 张图片, {num_classes} 个类别")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"日志文件: {LOG_FILE}，打开 http://localhost:8000 查看实时监控")
    print("-" * 50)

    log_data = init_log()
	# 确保必要的键存在
    log_data.setdefault("layer_weights", {})
    log_data.setdefault("layer_grads", {})
    write_log(log_data)

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(N)
        total_loss = 0.0
        epoch_start = time.time()

        total_box = 0.0
        total_cls = 0.0
        total_obj = 0.0
        batch_grad_avgs = {}

        last_print_time = time.time()

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, N)
            batch_indices = indices[start:end]

            batch_images = np.stack([dataset[i]['image'] for i in batch_indices], axis=0)
            batch_targets = [{'boxes': dataset[i]['boxes'], 'classes': dataset[i]['classes']}
                             for i in batch_indices]

            pred, caches = forward_yolo(batch_images, weights)
            loss,loss_comp = yolo_loss(pred, batch_targets, num_classes)
            total_loss += loss

            box_loss = loss_comp["box"]
            cls_loss = loss_comp["cls"]
            obj_loss = loss_comp["obj"]
            total_box += box_loss
            total_cls += cls_loss
            total_obj += obj_loss

            dloss = d_yolo_loss(pred, batch_targets, num_classes)
            grads = backward_yolo(dloss, caches)

            for key in grads:
                grad_abs_mean = np.abs(grads[key]).mean()
                if key not in batch_grad_avgs:
                    batch_grad_avgs[key] = []
                batch_grad_avgs[key].append(grad_abs_mean)

            for key in weights:
                grads[key] = np.clip(grads[key], -1.0, 1.0)
                weights[key] -= lr * grads[key]
                weights[key] = np.clip(weights[key], -5.0, 5.0)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                current_time = time.time()
                batch_time_total = current_time - last_print_time
                last_print_time = current_time
                print(f"Epoch {epoch}/{epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {loss:.4f}, Time: {batch_time_total:.2f}s")

        avg_loss = total_loss / num_batches
        avg_box = total_box / num_batches
        avg_cls = total_cls / num_batches
        avg_obj = total_obj / num_batches
        loss_components = {"box": avg_box, "cls": avg_cls, "obj": avg_obj}

        weight_avgs = {key: np.abs(weights[key]).mean() for key in weights}

        grad_avgs_epoch = {}
        for key, vals in batch_grad_avgs.items():
            grad_avgs_epoch[key] = np.mean(vals)

        epoch_time = time.time() - epoch_start

        log_data["epochs"].append(epoch)
        log_data["losses"].append(float(avg_loss))
        log_data["loss_components"]["box"].append(float(avg_box))
        log_data["loss_components"]["cls"].append(float(avg_cls))
        log_data["loss_components"]["obj"].append(float(avg_obj))
        log_data["learning_rates"].append(float(lr))
        log_data["times"].append(float(epoch_time))

        for key, val in weight_avgs.items():
            if key not in log_data["layer_weights"]:
                log_data["layer_weights"][key] = []
            log_data["layer_weights"][key].append(float(val))

        for key, val in grad_avgs_epoch.items():
            if key not in log_data["layer_grads"]:
                log_data["layer_grads"][key] = []
            log_data["layer_grads"][key].append(float(val))  # 直接记录梯度绝对值平均

        write_log(log_data)

        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        print("-" * 50)

    return weights, num_classes

def main():
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'
    IMG_SIZE = 416
    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 0.0001

    weights, num_classes = train_yolo(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    )

    np.savez('yolo_weights.npz', **weights)
    print("权重已保存到 yolo_weights.npz")

    # 测试单张图片
    dataset, _ = load_yolo_dataset(DATA_DIR, IMG_SIZE)
    sample = dataset[1]
    img = sample['image'][np.newaxis, ...]
    pred, _ = forward_yolo(img, weights)
    dets = decode_predictions(pred, img_size=IMG_SIZE, conf_thresh=0.3)
    img_display = (sample['image'].transpose(1, 2, 0) * 255).astype(np.uint8)
    visualize_detections(img_display, dets[0], title='YOLO Detection')


if __name__ == "__main__":
    main()
