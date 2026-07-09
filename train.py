# coding: utf-8
# 导入所有层函数
from funcs import (
    conv, d_conv,
    sigmoid, d_sigmoid,
	relu,d_relu,
    avgpool, d_avgpool,
    flatten, d_flatten,
    linear, d_linear,
    softmax, d_softmax,
    softmax_cross_entropy, d_softmax_cross_entropy,
	dropout,d_dropout
)
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb


def load_mnist_data(pkl_path):
    """加载 MNIST 数据"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # 数据格式: (train_data, val_data, test_data)
    train_data, val_data, test_data = data
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # 确保数据形状为 (N, C, H, W)
    if X_train.ndim == 2:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_val = X_val.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
    
    # 归一化到 [0, 1]
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # 如果像素值在 [0, 255]，归一化
    if X_train.max() > 1:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0
    
    # 转换为 (N, C, H, W)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_val = X_val.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"验证集: {X_val.shape}, 标签: {y_val.shape}")
    print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_mnist(X, y):
    """预处理：Resize 到 32x32 并添加 padding"""
    N = X.shape[0]
    X_32 = np.zeros((N, 1, 32, 32), dtype=np.float32)
    # 将 28x28 放到 32x32 的中心（padding 2）
    X_32[:, :, 2:30, 2:30] = X
    return X_32, y

def init_weights():
    """初始化 LeNet-5 权重 - Xavier 初始化"""
    weights = {}
    
    def xavier_init(shape):
        if len(shape) == 4:  # 卷积层
            fan_in = shape[1] * shape[2] * shape[3]
            fan_out = shape[0] * shape[2] * shape[3]
        else:  # 全连接层
            fan_in, fan_out = shape[0], shape[0]
        std = np.sqrt(2.0 / (fan_in))
        return np.random.randn(*shape).astype(np.float32) * std
    
    weights['conv1_w'] = xavier_init((6, 1, 5, 5))
    weights['conv1_b'] = np.zeros(6, dtype=np.float32)
    
    weights['conv2_w'] = xavier_init((16, 6, 5, 5))
    weights['conv2_b'] = np.zeros(16, dtype=np.float32)
    
    weights['conv3_w'] = xavier_init((120, 16, 5, 5))
    weights['conv3_b'] = np.zeros(120, dtype=np.float32)
    
    weights['fc1_w'] = xavier_init((120, 84))
    weights['fc1_b'] = np.zeros(84, dtype=np.float32)
    
    weights['fc2_w'] = xavier_init((84,10))
    weights['fc2_b'] = np.zeros(10, dtype=np.float32)
    
    return weights


def forward_pass(x, weights, training=True):
    """LeNet-5 前向传播"""
    caches = {}
    
    # 1. Conv1: 32x32x1 -> 28x28x6
    out, x_pad, (H_out, W_out) = conv(x, weights['conv1_w'], weights['conv1_b'], stride=1, padding=0)
    caches['conv1'] = (x, weights['conv1_w'], weights['conv1_b'], 1, 0, x_pad, H_out, W_out)
    
    # 2. Sigmoid
    out = relu(out)
    caches['relu1'] = out
    
    # 3. Pool1: 28x28x6 -> 14x14x6
    out, cache = avgpool(out, kernel_size=2, stride=2, padding=0)
    caches['pool1'] = cache
    
    # 4. Conv2: 14x14x6 -> 10x10x16
    conv2_input=out
    out, x_pad, (H_out, W_out) = conv(out, weights['conv2_w'], weights['conv2_b'], stride=1, padding=0)
    caches['conv2'] = (conv2_input, weights['conv2_w'], weights['conv2_b'], 1, 0, x_pad, H_out, W_out)
    
    # 5. Sigmoid
    out = relu(out)
    caches['relu2'] = out
    
    # 6. Pool2: 10x10x16 -> 5x5x16
    out, cache = avgpool(out, kernel_size=2, stride=2, padding=0)
    caches['pool2'] = cache
    
    # 7. Conv3: 5x5x16 -> 1x1x120
    conv3_input = out
    out, x_pad, (H_out, W_out) = conv(out, weights['conv3_w'], weights['conv3_b'], stride=1, padding=0)
    caches['conv3'] = (conv3_input, weights['conv3_w'], weights['conv3_b'], 1, 0, x_pad, H_out, W_out)
    
    # 8. Sigmoid
    out = relu(out)
    caches['relu3'] = out
    
    # 9. Flatten: 120 -> 120
    #out, cache = flatten(out)
    #caches['flatten'] = cache
    caches['flatten'] = out.shape
    out = out.reshape(out.shape[0],-1)
    
    # 10. FC1: 120 -> 84
    out, cache = linear(out, weights['fc1_w'], weights['fc1_b'])
    caches['fc1'] = cache
    
    # 11. Sigmoid
    out = relu(out)
    caches['relu4'] = out

    # ========== 插入 Dropout ==========
    out, cache = dropout(out, keep_prob=0.5, training=training)
    caches['dropout1'] = cache    

    # 12. FC2: 84 -> 10
    out, cache = linear(out, weights['fc2_w'], weights['fc2_b'])
    caches['fc2'] = cache
    
    return out, caches


def backward_pass(dout, caches, weights):
    """LeNet-5 反向传播"""
    grads = {}
    # 12. FC2 backward
    dx, dw, db = d_linear(dout, caches['fc2'])
    grads['fc2_w'] = dw
    grads['fc2_b'] = db

    # ========== Dropout 反向 ==========
    dx = d_dropout(dx, caches['dropout1']) 

    # 11. Sigmoid backward
    dx = dx * d_relu(None, caches['relu4'])
    
    # 10. FC1 backward
    dx, dw, db = d_linear(dx, caches['fc1'])
    grads['fc1_w'] = dw
    grads['fc1_b'] = db
    # 9. Flatten backward:
    flatten_shape = caches['flatten']
    dx = dx.reshape(flatten_shape)
    
	# 8. Sigmoid backward
    dx = dx * d_relu(None, caches['relu3'])
    
    # 7. Conv3 backward
    x, w, b, stride, padding, x_pad, H_out, W_out = caches['conv3']
    dx, dw, db = d_conv(dx, x, w, stride, padding, x_pad, H_out, W_out)
    grads['conv3_w'] = dw
    grads['conv3_b'] = db
    
    # 6. Pool2 backward
    dx = d_avgpool(dx, caches['pool2'])
    # 5. Sigmoid backward
    dx = dx * d_relu(None, caches['relu2'])
    
    # 4. Conv2 backward
    x, w, b, stride, padding, x_pad, H_out, W_out = caches['conv2']
    dx, dw, db = d_conv(dx, x, w, stride, padding, x_pad, H_out, W_out)
    grads['conv2_w'] = dw
    grads['conv2_b'] = db
    
    # 3. Pool1 backward
    dx = d_avgpool(dx, caches['pool1'])
    # 2. Sigmoid backward
    dx = dx * d_relu(None, caches['relu1'])
    
    
    # 1. Conv1 backward
    x, w, b, stride, padding, x_pad, H_out, W_out = caches['conv1']
    dx, dw, db = d_conv(dx, x, w, stride, padding, x_pad, H_out, W_out)
    grads['conv1_w'] = dw
    grads['conv1_b'] = db
    
    return grads


def compute_loss_and_gradients(x, y, weights):
    """计算损失和梯度"""
    # 前向
    logits, caches = forward_pass(x, weights,training=True)
    
    # Softmax + Cross Entropy Loss
    loss, ce_cache = softmax_cross_entropy(logits, y)
    
    # 反向
    dlogits = d_softmax_cross_entropy(ce_cache)
    grads = backward_pass(dlogits, caches, weights)
    
    return loss, grads, logits


def train_step(x, y, weights, learning_rate):
    """单步训练"""
    loss, grads, logits = compute_loss_and_gradients(x, y, weights)    
	# 梯度裁剪
    for key in grads.keys():
        grads[key] = np.clip(grads[key], -5.0, 5.0)
    # 更新权重
    for key in weights.keys():
        weights[key] -= learning_rate * grads[key]
    
    # 计算准确率
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    
    return loss, acc


def evaluate(x, y, weights):
    """评估模型"""
    logits, _ = forward_pass(x, weights, training=False)
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    return acc


def train_mnist(pkl_path, batch_size=64, epochs=10, learning_rate=0.001, verbose=True,num_samples=100):
    """训练 LeNet-5 在 MNIST 上"""
    
    # 加载数据
    print("加载 MNIST 数据...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data(pkl_path)
    # ========== 取少量数据用于快速测试 ==========
    if num_samples is not None:
        print(f"⚠️ 使用前 {num_samples} 张图片进行快速测试...")
        X_train = X_train[:num_samples]
        y_train = y_train[:num_samples]
        X_val = X_val[:min(num_samples // 5, len(X_val))]
        y_val = y_val[:min(num_samples // 5, len(y_val))]
        X_test = X_test[:min(num_samples // 5, len(X_test))]
        y_test = y_test[:min(num_samples // 5, len(y_test))]
    # 预处理：Resize 到 32x32
    print("预处理数据 (Resize 到 32x32)...")
    X_train, y_train = preprocess_mnist(X_train, y_train)
    X_val, y_val = preprocess_mnist(X_val, y_val)
    X_test, y_test = preprocess_mnist(X_test, y_test)
    # 初始化权重
    print("初始化权重...")
    weights = init_weights()
    
    # 训练参数
    N = X_train.shape[0]
    num_batches = N // batch_size
    
    train_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n开始训练...")
    print(f"训练集: {N} 张图片")
    print(f"Batch size: {batch_size}")
    print(f"总批次: {num_batches}")
    print(f"Epochs: {epochs}")
    print(f"学习率: {learning_rate}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Shuffle 数据
        perm = np.random.permutation(N)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        epoch_loss = 0
        epoch_acc = 0
        
        start_time = time.time()
        
        # 批次训练
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)
            
            batch_x = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]
            
            loss, acc = train_step(batch_x, batch_y, weights, learning_rate)
            
            epoch_loss += loss
            epoch_acc += acc
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {i+1}/{num_batches}, Loss: {loss:.4f}, Acc: {acc:.4f}")
        
        # 计算平均损失和准确率
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        # 验证集评估
        val_acc = evaluate(X_val, y_val, weights)
        
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        val_accs.append(val_acc)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {elapsed:.2f}s")
        print("-" * 60)
    
    # 测试集评估
    test_acc = evaluate(X_test, y_test, weights)
    print(f"\n测试集准确率: {test_acc:.4f}")
    
    return weights, train_losses, train_accs, val_accs, test_acc


def plot_training_curves(train_losses, train_accs, val_accs):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc')
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()


def predict(X, weights):
    """预测"""
    logits, _ = forward_pass(X, weights, training=False)
    pred = np.argmax(logits, axis=1)
    return pred


def visualize_predictions(X, y_true, weights, num_samples=10):
    """可视化预测结果"""
    # 预处理
    X_32, _ = preprocess_mnist(X, y_true)
    
    # 预测
    pred = predict(X_32[:num_samples], weights)
    
    # 显示
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = X[i, 0]  # 原始 28x28
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {y_true[i]}, Pred: {pred[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    # 训练参数
    PKL_PATH = './mnist.pkl'  # 你的 MNIST pkl 文件路径
    BATCH_SIZE =64
    EPOCHS = 20
    LEARNING_RATE = 0.05
    
    # 训练
    weights, train_losses, train_accs, val_accs, test_acc = train_mnist(
        pkl_path=PKL_PATH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        verbose=True,
		num_samples=1000
    )
    
    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_accs)
    
    # 加载测试集进行可视化
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    _, _, test_data = data
    X_test, y_test = test_data
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
    if X_test.max() > 1:
        X_test = X_test / 255.0
    
    visualize_predictions(X_test, y_test, weights, num_samples=10)
    
    print(f"\n训练完成!")
    print(f"最终测试集准确率: {test_acc:.4f}")
