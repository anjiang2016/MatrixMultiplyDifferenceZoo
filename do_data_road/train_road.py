"""
train_road.py
在 data_road 数据集上训练分割模型（裁剪 375×375 中心区域，缩放到 32×32）
"""
import os
import sys
# 将项目根目录添加到搜索路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# ============================================================
# 导入你的模块
# ============================================================
from funcs import conv, d_conv, relu, d_relu, sigmoid, d_sigmoid, avgpool, d_avgpool
from funcs import BACKEND, C_EXT_AVAILABLE
from fenge_road import (
    init_seg_weights, forward_seg, backward_seg,
    dice_loss, d_dice_loss,
    binary_cross_entropy, d_binary_cross_entropy,
    compute_iou, generate_seg_labels
)
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



# ============================================================
# 1. 数据加载（裁剪中间 375×375 + 缩放 32×32）
# ============================================================
def load_data_road(data_dir='data_road', split='training', target_size=32):
    image_dir = os.path.join(data_dir, split, 'image_2')
    gt_dir = os.path.join(data_dir, split, 'gt_image_2')

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    print(f"找到 {len(image_files)} 个图像文件")

    X = []
    y = []

    crop_size = 375
    x_start = (1242 - crop_size) // 2
    x_end = x_start + crop_size

    for f in image_files:
        img_path = os.path.join(image_dir, f)

        name, ext = os.path.splitext(f)
        parts = name.split('_', 1)
        if len(parts) == 2:
            gt_name = f"{parts[0]}_road_{parts[1]}{ext}"
        else:
            gt_name = f"{name}_road{ext}"
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            continue

        # 加载图像
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img_crop = img[:, x_start:x_end]
        img_resized = Image.fromarray(img_crop).resize((target_size, target_size))
        img_resized = np.array(img_resized).astype(np.float32) / 255.0
        img_resized = img_resized.transpose(2, 0, 1)
        X.append(img_resized)

        # ===== 加载标注（提取粉色 = 道路） =====
        gt = Image.open(gt_path).convert('RGB')
        gt = np.array(gt)
        gt_crop = gt[:, x_start:x_end]

        # 粉色道路：R>200, G<100, B>200
        road_mask = (gt_crop[:, :, 0] > 200) & (gt_crop[:, :, 1] < 100) & (gt_crop[:, :, 2] > 200)
        gt_binary = road_mask.astype(np.float32)

        gt_resized = Image.fromarray((gt_binary * 255).astype(np.uint8)).resize(
            (target_size, target_size), Image.NEAREST
        )
        gt_resized = np.array(gt_resized).astype(np.float32) / 255.0
        gt_resized = (gt_resized > 0.5).astype(np.float32)
        gt_resized = gt_resized.reshape(1, target_size, target_size)
        y.append(gt_resized)

    X = np.array(X)
    y = np.array(y)
    print(f"加载完成: {len(X)} 张有标注的图像, 输入形状: {X.shape}, 标签形状: {y.shape}")
    return X, y

# ============================================================
# 2. 数据增强
# ============================================================
def augment_batch(X, Y):
    """
    批量数据增强（在线）
    X: (B, 3, H, W) 图像
    Y: (B, 1, H, W) 标签
    """
    B = X.shape[0]
    
    # 水平翻转（随机一半样本）
    flip_mask = np.random.rand(B) > 0.5
    for i in range(B):
        if flip_mask[i]:
            X[i] = np.flip(X[i], axis=2).copy()
            Y[i] = np.flip(Y[i], axis=2).copy()
    
    # 亮度调整（仅对图像）
    brightness = 0.85 + 0.3 * np.random.rand(B, 1, 1, 1)
    X = X * brightness
    X = np.clip(X, 0, 1)
    
    return X, Y


# ============================================================
# 3. 训练函数
# ============================================================
def train_road(
    data_dir='data_road',
    epochs=50,
    batch_size=16,
    lr=0.001,
    target_size=32,
    loss_type='dice',           # 'dice' 或 'bce'
    pos_weight=10.0,
    save_interval=10,
    save_dir='models_road',
    resume_from=None
):
    """
    在 data_road 上训练分割模型
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # ---- 加载数据 ----
    print("加载 data_road ...")
    X_full, y_full = load_data_road(data_dir, 'training', target_size)
    # 划分训练/验证集（8:2）
    N = len(X_full)
    indices = np.random.permutation(N)
    split = int(N * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]
    
    print(f"训练集: {len(X_train)} 张, 验证集: {len(X_val)} 张")
    
    # ---- 初始化模型 ----
    print("初始化权重...")
    weights = init_seg_weights()  # 注意：输入通道为 3
    start_epoch = 0
    
    if resume_from and os.path.exists(resume_from):
        loaded = np.load(resume_from, allow_pickle=True)
        for key in weights:
            if key in loaded:
                weights[key] = loaded[key]
        # 如果文件中保存了 iou 信息
        if 'best_iou' in loaded:
            best_iou = float(loaded['best_iou'])
            print(f"恢复最佳 IoU: {best_iou:.4f}")
        else:
            best_iou = 0.0
    else:
        best_iou = 0.0
        print(f"从 {resume_from} 恢复权重")
        # 如果文件名包含 epoch 信息，可以解析
        # start_epoch = int(resume_from.split('_')[-1].split('.')[0]) + 1
    
    # ---- 训练循环 ----
    N_train = len(X_train)
    num_batches = max(1, N_train // batch_size)
    
    train_losses = []
    val_iou_history = []
    
    print(f"\n开始训练...")
    print(f"设备: {'C 扩展' if C_EXT_AVAILABLE else 'NumPy'} (BACKEND={BACKEND})")
    print(f"损失: {loss_type}, 学习率: {lr}, Epochs: {epochs}")
    print("-" * 60)
    
    for epoch in range(start_epoch, epochs):
        # 打乱数据
        perm = np.random.permutation(N_train)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        epoch_loss = 0
        start_time = time.time()
        
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, N_train)
            
            batch_x = X_shuffled[start:end].copy()
            batch_y = y_shuffled[start:end].copy()
            
            # ---- 数据增强 ----
            batch_x, batch_y = augment_batch(batch_x, batch_y)
            
            # ---- 前向传播 ----
            pred, caches = forward_seg(batch_x, weights, training=True)
            
            # ---- 损失函数 ----
            if loss_type == 'dice':
                loss, cache = dice_loss(pred, batch_y)
                dloss = d_dice_loss(cache)
            else:  # bce
                loss, cache = binary_cross_entropy(pred, batch_y, pos_weight=pos_weight)
                dloss = d_binary_cross_entropy(cache)
            
            # ---- 反向传播 ----
            grads = backward_seg(dloss, caches, weights)
            
            # ---- 梯度裁剪 ----
            for key in grads.keys():
                grads[key] = np.clip(grads[key], -1.0, 1.0)
            
            # ---- 更新权重 ----
            for key in weights.keys():
                weights[key] -= lr * grads[key]
            
            epoch_loss += loss
        
        # ---- 验证 ----
        pred_val, _ = forward_seg(X_val, weights, training=False)
        val_loss, _ = dice_loss(pred_val, y_val)  # 用 Dice 评估
        val_iou = compute_iou(pred_val, y_val)
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        val_iou_history.append(val_iou)
        
        # ---- 保存最佳模型 ----
        if val_iou > best_iou:
            best_iou = val_iou
            best_path = os.path.join(save_dir, 'road_best.npz')
            np.savez(best_path, **weights,best_iou=best_iou)
            print(f"✅ 保存最佳模型: IoU={best_iou:.4f}")
        
        # ---- 定期保存 ----
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f'road_epoch_{epoch+1:03d}.npz')
            np.savez(checkpoint_path, **weights)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Time: {elapsed:.2f}s")
        print("-" * 60)
    
    print(f"\n训练完成！最佳 IoU: {best_iou:.4f}")
    
    return weights, train_losses, val_iou_history


# ============================================================
# 4. 可视化结果
# ============================================================
def visualize_road_results(X, y_true, weights, num_samples=4, threshold=0.5,save_path=None):
    """
    可视化分割结果
    """
    pred, _ = forward_seg(X[:num_samples], weights, training=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, num_samples*3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # 原图 (RGB, 32x32)
        img = X[i].transpose(1, 2, 0)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        # 真实标签
        axes[i, 1].imshow(y_true[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # 预测
        axes[i, 2].imshow(pred[i, 0]>threshold, cmap='gray', vmin=0, vmax=1)
        iou = compute_iou(pred[i:i+1], y_true[i:i+1])
        axes[i, 2].set_title(f'Pred (IoU={iou:.3f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ============================================================
# 5. 主入口
# ============================================================
if __name__ == "__main__":
    # 设置后端（通过环境变量控制）
    # 运行前可设置: export BACKEND=numpy 或 export OMP_NUM_THREADS=16
    
    # 训练
    weights, losses, ious = train_road(
        data_dir='/Users/zhaomingming/data_sets/data_road',
        epochs=30,
        batch_size=64,
        lr=0.005,
        target_size=32,
        loss_type='dice',          # 先用 Dice Loss
        save_interval=10,
        save_dir='models_road',
        resume_from='models_road/road_best.npz'  # 或者 road_epoch_005.npze
    )
    
    # 加载测试数据（取验证集展示）
    X_full, y_full = load_data_road('/Users/zhaomingming/data_sets/data_road', 'training', 32)
	# 替换 train_test_split
    N = len(X_full)
    indices = np.random.permutation(N)
    split = int(N * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train, y_val = y_full[train_idx], y_full[val_idx]
    
    # 可视化
    visualize_road_results(X_val, y_val, weights, num_samples=4, save_path='road_results.png')
    
    print("\n训练完成！")
