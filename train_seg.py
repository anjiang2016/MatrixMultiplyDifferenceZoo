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
def load_mnist_data(pkl_path):
    """加载 MNIST pkl 文件"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    train_data, val_data, test_data = data
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    if X_train.ndim == 2:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_val = X_val.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
    
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    if X_train.max() > 1:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_mnist(X, y):
    """Resize 28x28 -> 32x32 (padding 2)"""
    # 如果 X 是 (N, 28, 28)，则添加通道维度
    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]  # (N, 1, 28, 28)
    N = X.shape[0]
    X_32 = np.zeros((N, 1, 32, 32), dtype=np.float32)
    X_32[:, :, 2:30, 2:30] = X
    return X_32, y


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
