import pickle
import numpy as np
import numpy as np
import os

def load_mnist_from_files(data_dir):
    """
    从 MNIST 的 .ubyte 文件加载数据
    data_dir 应包含 train/ 和 test/ 两个子文件夹
    """
    def read_images(filepath):
        with open(filepath, 'rb') as f:
            # 前 4 个字节是 magic number，接下来 4 个是图片数量，然后是行数和列数
            magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num, rows, cols).astype(np.float32) / 255.0

    def read_labels(filepath):
        with open(filepath, 'rb') as f:
            magic, num = np.frombuffer(f.read(8), dtype='>i4')
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    # 读取训练集
    X_train = read_images(os.path.join(data_dir, 'train', 'train-images-idx3-ubyte'))
    y_train = read_labels(os.path.join(data_dir, 'train', 'train-labels-idx1-ubyte'))

    # 读取测试集
    X_test = read_images(os.path.join(data_dir, 'test', 't10k-images-idx3-ubyte'))
    y_test = read_labels(os.path.join(data_dir, 'test', 't10k-labels-idx1-ubyte'))

    return (X_train, y_train), (X_test, y_test)

# 使用示例
data_dir = '/Users/zhaomingming/data_sets/MNIST_Data'  # 指向解压后的根目录
(X_train, y_train), (X_test, y_test) = load_mnist_from_files(data_dir)

print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
# 先用上面的函数加载
(X_train, y_train), (X_test, y_test) = load_mnist_from_files('/Users/zhaomingming/data_sets/MNIST_Data')

# 构建你需要的 (train, val, test) 结构（示例）
# 假设你之前用 5000 张做验证，5000 张做测试
val_size = 5000
X_val = X_train[:val_size]
y_val = y_train[:val_size]
X_train_new = X_train[val_size:]
y_train_new = y_train[val_size:]

# 构造与之前兼容的 tuple
data = ((X_train_new, y_train_new), (X_val, y_val), (X_test, y_test))

with open('/Users/zhaomingming/data_sets/mnist.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✅ mnist.pkl 已生成！")
print(f"训练集: {X_train_new.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
