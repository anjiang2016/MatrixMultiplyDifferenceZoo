"""
train_seg.py - 分割模型训练脚本
依赖 funcs.py（基础层）和 fenge.py（分割专用模块）
"""
from funcs import (
    conv, d_conv,
    relu, d_relu,
    sigmoid, d_sigmoid,
    avgpool, d_avgpool,
    linear, d_linear,
    flatten, d_flatten,
    softmax, d_softmax,
    softmax_cross_entropy, d_softmax_cross_entropy
)

from fenge import (
    upsample_bilinear, d_upsample_bilinear,
    init_seg_weights,
    forward_seg, backward_seg,
    binary_cross_entropy, d_binary_cross_entropy,
    binary_cross_entropy_with_dice, d_binary_cross_entropy_with_dice,
    dice_loss, d_dice_loss,
    save_seg_model,load_seg_model,list_saved_models,os,
    generate_seg_labels, compute_iou
)
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt


# ============================================================
# 数据加载与预处理
# ============================================================
import os
import numpy as np
from PIL import Image

def load_data_road_crop(data_dir='data_road', split='training', target_size=32):
    """
    加载 data_road，从中间裁剪 375×375 区域，再缩放到 target_size×target_size
    """
    image_dir = os.path.join(data_dir, split, 'image_2')
    gt_dir = os.path.join(data_dir, split, 'gt_image_2')
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    X = []
    y = []
    
    # 原图宽 1242，高 375，裁出中间 375×375
    crop_size = 375
    x_start = (1242 - crop_size) // 2  # 433
    x_end = x_start + crop_size         # 808
    
    for f in image_files:
        # ===== 加载图像 =====
        img = Image.open(os.path.join(image_dir, f)).convert('RGB')
        img = np.array(img)  # (375, 1242, 3)
        
        # 裁剪中间 375×375
        img_crop = img[:, x_start:x_end]  # (375, 375, 3)
        
        # 缩放到 target_size × target_size
        img_resized = Image.fromarray(img_crop).resize((target_size, target_size))
        img_resized = np.array(img_resized).astype(np.float32) / 255.0
        img_resized = img_resized.transpose(2, 0, 1)  # (3, target_size, target_size)
        X.append(img_resized)
        
        # ===== 加载标注（如果存在）=====
        gt_path = os.path.join(gt_dir, f)
        if os.path.exists(gt_path):
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)  # (375, 1242)
            
            # 同样的裁剪
            gt_crop = gt[:, x_start:x_end]  # (375, 375)
            
            # 缩放到 target_size × target_size（标签用最近邻，保持二值性）
            gt_resized = Image.fromarray(gt_crop).resize((target_size, target_size), Image.NEAREST)
            gt_resized = np.array(gt_resized).astype(np.float32) / 255.0
            gt_resized = (gt_resized > 0.5).astype(np.float32)
            gt_resized = gt_resized.reshape(1, target_size, target_size)  # (1, target_size, target_size)
            y.append(gt_resized)
        else:
            y.append(None)
    
    # 过滤掉没有标注的图片（如果有）
    valid_idx = [i for i, yi in enumerate(y) if yi is not None]
    X = np.array([X[i] for i in valid_idx])
    y = np.array([y[i] for i in valid_idx])
    
    print(f"加载完成: {len(X)} 张图像, 输入形状: {X.shape}, 标签形状: {y.shape}")
    return X, y

# ============================================================
# 训练函数
# ============================================================
def train_seg_mnist(pkl_path, num_samples=100, batch_size=16, epochs=30, lr=0.001, verbose=True,resume_from=None,save_interval=5,save_dir='models'):
    """训练分割模型"""
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data(pkl_path)
    
    X_train = X_train[:num_samples]
    X_val = X_val[:max(1, num_samples // 5)]
    X_test = X_test[:max(1, num_samples // 5)]
    
    X_train, _ = preprocess_mnist(X_train, y_train)
    X_val, _ = preprocess_mnist(X_val, y_val)
    X_test, _ = preprocess_mnist(X_test, y_test)
    
    y_seg_train = generate_seg_labels(X_train)
    y_seg_val = generate_seg_labels(X_val)
    y_seg_test = generate_seg_labels(X_test)
    # ========== 初始化或恢复权重 ==========
    if resume_from and os.path.exists(resume_from):
        weights, metadata = load_seg_model(resume_from)
        start_epoch = metadata.get('epoch', 0) + 1
        print(f"📌 从 Epoch {start_epoch} 继续训练")
        
        # 可以选择恢复优化器状态（如果需要）
        # optimizer_state = metadata.get('optimizer_state', None)
    else:
        weights = init_seg_weights()
        start_epoch = 0
        print("📌 从头开始训练")
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    N = X_train.shape[0]
    num_batches = max(1, N // batch_size)
    
    print(f"训练分割模型，{N} 张图片，{num_batches} 个 batch")
    print(f"总 Epoch: {epochs}，学习率: {lr}")
    print("-" * 60)
    
    best_iou = 0.0

    for epoch in range(epochs):
        perm = np.random.permutation(N)
        X_shuffled = X_train[perm]
        y_shuffled = y_seg_train[perm]
        
        epoch_loss = 0
        start_time = time.time()
        # 每个 epoch 衰减
        lr = 0.01  * (0.95 ** epoch) 
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, N)
            batch_x = X_shuffled[start:end]
            batch_y = y_shuffled[start:end]
            
            pred, caches = forward_seg(batch_x, weights, training=True)
            loss, ce_cache = binary_cross_entropy(pred, batch_y,pos_weight=9.0)
            dloss = d_binary_cross_entropy(ce_cache)
            #loss, dice_cache = dice_loss(pred, batch_y)
            #dloss = d_dice_loss(dice_cache)
            grads = backward_seg(dloss, caches, weights)
            
            for key in grads.keys():
                grads[key] = np.clip(grads[key], -1.0, 1.0)
             
            for key in weights.keys():
                weights[key] -= lr * grads[key]
            
            epoch_loss += loss
        
        pred_val, _ = forward_seg(X_val, weights, training=False)
        #val_loss, _ = binary_cross_entropy(pred_val, y_seg_val)
        val_loss, _ = dice_loss(pred_val, y_seg_val)
        iou_val = compute_iou(pred_val, y_seg_val)
    
        # 保存最佳模型
        if iou_val > best_iou:
            best_iou = iou_val
            best_path = os.path.join(save_dir, f'seg_model_best.npz')
            save_seg_model(weights, best_path, epoch, epoch_loss/num_batches, iou_val)
        
        # 定期保存
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f'seg_model_epoch_{epoch+1:03d}.npz')
            save_seg_model(weights, checkpoint_path, epoch+1, epoch_loss/num_batches, iou_val)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/num_batches:.4f}, Val Loss: {val_loss:.4f}, IoU: {iou_val:.4f}, Time: {time.time()-start_time:.2f}s")
    
    pred_test, _ = forward_seg(X_test, weights, training=False)
    test_iou = compute_iou(pred_test, y_seg_test)
    print(f"\n测试集 IoU: {test_iou:.4f}")
    
    return weights, (X_test, y_seg_test)


def visualize_segmentation(X, y_true, weights, num_samples=5, threshold=0.5, save_path=None):
    """可视化分割结果"""
    pred, _ = forward_seg(X[:num_samples], weights, training=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, num_samples*3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # 原图
        axes[i, 0].imshow(X[i, 0], cmap='gray')
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        # 真实标签（已经是 0/1）
        axes[i, 1].imshow(y_true[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # ========== 预测图：用阈值二值化 ==========
        pred_binary = (pred[i, 0] > threshold).astype(np.float32)
        axes[i, 2].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
        iou = compute_iou(pred_binary, y_true[i, 0])  # 注意这里用二值化后的计算 IoU
        axes[i, 2].set_title(f'Pred (IoU={iou:.3f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    PKL_PATH = '/Users/zhaomingming/data_sets/mnist/mnist.pkl'
    
    weights, (X_test, y_seg_test) = train_seg_mnist(
        pkl_path=PKL_PATH,
        num_samples=2000,
        batch_size=32,
        epochs=100,
        lr=0.01,
        verbose=True,
        resume_from='models/seg_model_best.npz',
		save_interval=5,
		save_dir='models'
    )
    
    visualize_segmentation(X_test, y_seg_test, weights, num_samples=5)
