"""
train_yolo.py - 训练 YOLO 检测器（纯 NumPy），集成实时网页监控
包含断点续训功能：自动加载历史最佳模型，并持续优化
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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
LOG_FILE = "training_log_class_weight.json"

def init_log():
    """初始化日志数据结构"""
    return {
        "epochs": [],
        "losses": [],
        "loss_components": {"box": [], "cls": [], "obj": []},
        "learning_rates": [],
        "times": [],
        "layer_weights": {},
        "layer_grads": {}
    }

def write_log(log_data):
    """写入 JSON 文件"""
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)
def compute_class_weights_effective(class_counts, num_classes, beta=0.999):
    """
    有效样本数加权，参考论文《Class-Balanced Loss》
    返回长度为 num_classes 的权重列表
    """
    weights = {}
    for cls_id, count in class_counts.items():
        weights[cls_id] = (1 - beta) / (1 - beta ** count)
    
    # 计算所有存在类别的权重均值
    mean_weight = sum(weights.values()) / len(weights) if weights else 1.0
    
    # 构建完整列表，缺失类别权重设为 1.0（或均值）
    class_weights_list = []
    for i in range(num_classes):
        if i in weights:
            class_weights_list.append(weights[i] / mean_weight)
        else:
            class_weights_list.append(1.0)  # 无样本类别权重设为 1.0
    return class_weights_list
def compute_class_weights(class_counts, num_classes, gamma=0.5):
    total = sum(class_counts.values())
    weights = {}
    for cls_id, count in class_counts.items():
        weights[cls_id] = (total / count) ** gamma
    mean_weight = sum(weights.values()) / len(weights) if weights else 1.0
    # 构建完整列表，缺失类别权重设为 1.0（或均值）
    class_weights_list = []
    for i in range(num_classes):
        if i in weights:
            class_weights_list.append(weights[i] / mean_weight)
        else:
            class_weights_list.append(1.0)  # 或 1.0 / mean_weight
    return class_weights_list
def compute_class_weights_inverse(class_counts, num_classes):
    """
    逆频率比例法：权重 = total / count，归一化后最大值拉伸到 1.0
    
    示例：
    class_counts = {0: 2, 1: 3, 2: 5}
    total = 10
    原始权重: [5, 10/3, 2] = [5, 3.33, 2]
    归一化: [5/10.33, 3.33/10.33, 2/10.33] = [0.484, 0.323, 0.194]
    最大值拉伸到 1.0: [0.484/0.484, 0.323/0.484, 0.194/0.484] = [1.0, 0.667, 0.4]
    """
    if not class_counts:
        return [1.0] * num_classes
    
    total = sum(class_counts.values())
    raw_weights = []
    for i in range(num_classes):
        if i in class_counts and class_counts[i] > 0:
            raw_weights.append(total / class_counts[i])
        else:
            raw_weights.append(1.0)
    
    # 归一化使总和为 1（或任意常数）
    sum_weights = sum(raw_weights)
    normalized = [w / sum_weights for w in raw_weights]
    
    # 最大值拉伸到 1.0
    max_w = max(normalized)
    final_weights = [w / max_w for w in normalized]
    
    return final_weights
def adam_step(params, grads, state, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    """
    Adam 优化器一步更新

    Args:
        params: 权重字典
        grads: 梯度字典
        state: 优化器状态，包含 't', 'm', 'v'
        lr: 学习率
        beta1: 一阶矩衰减率
        beta2: 二阶矩衰减率
        eps: 防止除零
        weight_decay: L2 正则化系数

    Returns:
        new_params, new_state
    """
    if 't' not in state:
        state['t'] = 0
    if 'm' not in state:
        state['m'] = {}
    if 'v' not in state:
        state['v'] = {}
    
    state['t'] += 1
    t = state['t']
    
    new_params = {}
    for key in params.keys():
        if key not in state['m']:
            state['m'][key] = np.zeros_like(params[key])
        if key not in state['v']:
            state['v'][key] = np.zeros_like(params[key])
        
        grad = grads[key] + weight_decay * params[key]
        
        state['m'][key] = beta1 * state['m'][key] + (1 - beta1) * grad
        state['v'][key] = beta2 * state['v'][key] + (1 - beta2) * (grad ** 2)
        
        m_hat = state['m'][key] / (1 - beta1 ** t)
        v_hat = state['v'][key] / (1 - beta2 ** t)
        
        new_params[key] = params[key] - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return new_params, state
# ============================================================
# 训练函数
# ============================================================
def train_yolo(data_dir, img_size=416, epochs=1, batch_size=2, lr=0.001,lr_step_size=10,lr_gamma=0.9,use_adam=True,beta1=0.9,beta2=0.999,weight_decay=0.0001):
    dataset, num_classes = load_yolo_dataset(data_dir, img_size)
	# 在 train_yolo 函数中，加载数据集后
    class_counts = Counter()
    for sample in dataset:
        for cls in sample['classes']:
            class_counts[int(cls)] += 1

    #class_weights = compute_class_weights_effective(class_counts,num_classes, beta=0.995)  # 或使用有效样本数方法
    #class_weights = compute_class_weights(class_counts,num_classes,gamma=2.5) 
    class_weights = compute_class_weights_inverse(class_counts,num_classes) 
    print("类别权重:", class_weights)

    weights = init_yolo_weights(num_classes)
    # 初始化 Adam 状态
    adam_state = None
    if use_adam:
        adam_state = {'t': 0, 'm': {}, 'v': {}}
    # ---------- 加载最佳模型（断点续训） ----------
    best_model_path = 'best_model.npz'
    best_loss = float('inf')
    start_epoch = 1  # 默认从第1轮开始

    if os.path.exists(best_model_path):
        best_data = np.load(best_model_path, allow_pickle=True)
        # 加载权重
        for key in weights:
            if key in best_data:
                weights[key] = best_data[key]
        # 加载最佳 Loss 和 epoch
        best_loss = float(best_data.get('best_loss', float('inf')))
        best_epoch = int(best_data.get('best_epoch', 0))
        start_epoch = best_epoch + 1
        print(f"✅ 加载最佳模型 (epoch {best_epoch}, loss {best_loss:.4f})，从 epoch {start_epoch} 继续训练")
        if use_adam and 'adam_state' in best_data:
            adam_state = best_data['adam_state'].item()
            print("✅ 恢复最佳模型的优化器状态")
    else:
        print("📌 未找到最佳模型，从头开始训练")

    N = len(dataset)
    num_batches = int(np.ceil(N / batch_size))
    print(f"数据集: {N} 张图片, {num_classes} 个类别")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"日志文件: {LOG_FILE}，打开 http://localhost:8000 查看实时监控")
    print("-" * 50)

    # ---------- 加载已有日志（保留历史记录） ----------
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            log_data = json.load(f)
        # 确保必要的键存在
        log_data.setdefault("layer_weights", {})
        log_data.setdefault("layer_grads", {})
        print(f"已加载历史日志，已有 {len(log_data['epochs'])} 条记录")
    else:
        log_data = init_log()
        log_data.setdefault("layer_weights", {})
        log_data.setdefault("layer_grads", {})

    # ---------- 训练循环 ----------
    for epoch in range(start_epoch, epochs + 1):
        # 计算当前学习率
        current_lr = lr * (lr_gamma ** ((epoch-start_epoch) // lr_step_size))

        # 打印学习率变化
        if epoch == start_epoch or (epoch - start_epoch) % lr_step_size == 0:
            print(f"Epoch {epoch+1}: Learning rate = {current_lr:.6f}")
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
            # 在损失计算时传入权重
            loss, loss_comp = yolo_loss(pred, batch_targets, num_classes, class_weights=class_weights)
            total_loss += loss

            box_loss = loss_comp["box"]
            cls_loss = loss_comp["cls"]
            obj_loss = loss_comp["obj"]
            total_box += box_loss
            total_cls += cls_loss
            total_obj += obj_loss

            dloss = d_yolo_loss(pred, batch_targets, num_classes)
            grads = backward_yolo(dloss, caches)

            # 记录梯度
            for key in grads:
                grad_abs_mean = np.abs(grads[key]).mean()
                if key not in batch_grad_avgs:
                    batch_grad_avgs[key] = []
                batch_grad_avgs[key].append(grad_abs_mean)

            # ---------- 更新权重（取消裁剪以加速过拟合） ----------
			            # 更新权重
            if use_adam:

                weights, adam_state = adam_step(
                    weights, grads, adam_state,
                    lr=current_lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay
                )
            else:
                # 原始 SGD 更新
                for key in weights:
                    weights[key] -= current_lr * grads[key]

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                current_time = time.time()
                batch_time_total = current_time - last_print_time
                last_print_time = current_time
                print(f"Epoch {epoch}/{epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {loss:.4f}, Time: {batch_time_total:.2f}s")

        # ---------- 计算 epoch 平均值 ----------
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

        # ---------- 记录日志 ----------
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
            log_data["layer_grads"][key].append(float(val))

        write_log(log_data)

        # ---------- 保存最佳模型 ----------
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_data = {**weights, 'best_loss': best_loss, 'best_epoch': best_epoch}
            if use_adam:
                best_data['adam_state'] = adam_state
            np.savez(best_model_path, **best_data)
            print(f"⭐ 新最佳模型保存 (epoch {epoch}, loss {best_loss:.4f})")

        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        print("-" * 50)

    # 最终权重保存（可选）
    np.savez('yolo_weights.npz', **weights)
    print("最终权重已保存到 yolo_weights.npz")
    return weights, num_classes


def main():
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'
    IMG_SIZE = 416
    EPOCHS =750        # 增加 epoch 以加速过拟合
    BATCH_SIZE = 32
    LR = 0.0003            # 增大学习率

    weights, num_classes = train_yolo(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    )

    # 测试单张图片（可注释掉以节省时间）
    dataset, _ = load_yolo_dataset(DATA_DIR, IMG_SIZE)
    sample = dataset[1]
    img = sample['image'][np.newaxis, ...]
    pred, _ = forward_yolo(img, weights)
    dets = decode_predictions(pred, img_size=IMG_SIZE, conf_thresh=0.4)
    img_display = (sample['image'].transpose(1, 2, 0) * 255).astype(np.uint8)
    visualize_detections(img_display, dets[0], title='YOLO Detection')


if __name__ == "__main__":
    main()
