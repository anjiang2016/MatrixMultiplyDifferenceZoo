import pickle
import matplotlib.pyplot as plt
import numpy as np
# 显示pkl格式的mnist数据集内的图片
# 加载数据
with open('/Users/zhaomingming/data_sets/mnist/mnist.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

print(f"数据类型: {type(data)}")
print(f"数据长度: {len(data)}")

# 解析三个数据集
train_data, val_data, test_data = data

# 提取图片和标签
X_train, y_train = train_data
X_val, y_val = val_data
X_test, y_test = test_data

print(f"\n训练集: {X_train.shape}, 标签: {y_train.shape}")
print(f"验证集: {X_val.shape}, 标签: {y_val.shape}")
print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")

# ========== 1. 显示训练集的前 10 张图片 ==========
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    img = X_train[i]
    
    # 如果是扁平数据 (784,)，reshape 为 28x28
    if img.ndim == 1 and len(img) == 784:
        img = img.reshape(28, 28)
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Label: {int(y_train[i])}', fontsize=12)
    axes[i].axis('off')

plt.suptitle('MNIST 训练集示例', fontsize=16)
plt.tight_layout()
plt.show()

# ========== 2. 统计标签分布 ==========
def show_label_distribution(labels, title):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{title}")
    for digit, count in zip(unique, counts):
        print(f"  数字 {int(digit)}: {count} 张 ({count/len(labels)*100:.1f}%)")

show_label_distribution(y_train, "训练集标签分布:")
show_label_distribution(y_val, "验证集标签分布:")
show_label_distribution(y_test, "测试集标签分布:")

# ========== 3. 显示每个数字的示例（修正版）==========
# 方式A: 显示 2 行 5 列，每个数字显示 2 张
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for digit in range(10):
    # 找到该数字的所有索引
    indices = np.where(y_train == digit)[0]
    # 只取前 1 张（因为我们是 2x5=10 个子图）
    idx = indices[0]
    img = X_train[idx]
    if img.ndim == 1 and len(img) == 784:
        img = img.reshape(28, 28)
    
    axes[digit].imshow(img, cmap='gray')
    axes[digit].set_title(f'数字: {int(y_train[idx])}', fontsize=14)
    axes[digit].axis('off')

plt.suptitle('每个数字的示例（各1张）', fontsize=16)
plt.tight_layout()
plt.show()

# ========== 4. 方式B: 显示每个数字的多个示例 ==========
# 创建一个 10x5 的网格（10个数字 × 5张图片）
fig, axes = plt.subplots(10, 5, figsize=(12, 20))

for digit in range(10):
    # 找到该数字的所有索引
    indices = np.where(y_train == digit)[0]
    # 取前 5 张
    for i in range(min(5, len(indices))):
        idx = indices[i]
        img = X_train[idx]
        if img.ndim == 1 and len(img) == 784:
            img = img.reshape(28, 28)
        
        axes[digit, i].imshow(img, cmap='gray')
        axes[digit, i].set_title(f'{int(y_train[idx])}', fontsize=10)
        axes[digit, i].axis('off')
    
    # 如果该数字不够5张，隐藏多余的子图
    for i in range(min(5, len(indices)), 5):
        axes[digit, i].axis('off')

plt.suptitle('每个数字的示例（各5张）', fontsize=16)
plt.tight_layout()
plt.show()

# ========== 5. 打印数据集统计信息 ==========
print(f"\n=== 数据集统计 ===")
print(f"总图片数: {len(X_train) + len(X_val) + len(X_test)}")
print(f"训练集: {len(X_train)} 张")
print(f"验证集: {len(X_val)} 张")
print(f"测试集: {len(X_test)} 张")
print(f"图片尺寸: {X_train[0].shape}")
print(f"像素范围: [{X_train.min():.2f}, {X_train.max():.2f}]")
