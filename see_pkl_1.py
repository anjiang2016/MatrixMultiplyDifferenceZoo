import pickle
import matplotlib.pyplot as plt
import numpy as np

# 方法1: 使用 latin1 编码（最常用）
with open('/Users/zhaomingming/data_sets/mnist/mnist.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 如果方法1不行，试试方法2: 使用 bytes
# with open('mnist.pkl', 'rb') as f:
#     data = pickle.load(f, encoding='bytes')

# 如果还不行，试试方法3: 使用 ASCII + 忽略错误
# with open('mnist.pkl', 'rb') as f:
#     data = pickle.load(f, encoding='ascii', errors='ignore')

# 查看数据结构
print(f"数据类型: {type(data)}")
print(f"数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")

# 如果是 tuple，查看内容和形状
if isinstance(data, tuple):
    print(f"tuple 长度: {len(data)}")
    for i, item in enumerate(data):
        print(f"第 {i} 个元素类型: {type(item)}")
        if isinstance(item, (np.ndarray, list)):
            print(f"  形状: {item.shape if hasattr(item, 'shape') else len(item)}")
        else:
            print(f"  内容: {item}")
    
    # 尝试显示图片（假设格式是 (X_train, y_train), (X_test, y_test)）
    if len(data) == 2 and isinstance(data[0], tuple):
        (X_train, y_train), (X_test, y_test) = data
        images = X_train
        labels = y_train
        print(f"\n训练集大小: {len(images)}")
        print(f"图片形状: {images[0].shape if len(images) > 0 else 'N/A'}")
        
        # 显示前 10 张
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))
        axes = axes.ravel()
        
        for i in range(min(10, len(images))):
            img = images[i]
            # 如果是扁平数据，reshape
            if img.ndim == 1 and len(img) == 784:
                img = img.reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {labels[i]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("\n数据格式不是预期的 (X_train, y_train), (X_test, y_test)")
        print("请根据上面的打印信息调整代码")

# 如果是 dict，查看键和值
elif isinstance(data, dict):
    print(f"字典键: {list(data.keys())}")
    for key, value in data.items():
        print(f"  {key}: {type(value)}")
        if isinstance(value, (np.ndarray, list)):
            print(f"    形状: {value.shape if hasattr(value, 'shape') else len(value)}")

# 如果是 list
elif isinstance(data, list):
    print(f"列表长度: {len(data)}")
    if len(data) > 0:
        print(f"第一个元素类型: {type(data[0])}")
        if isinstance(data[0], (np.ndarray, list)):
            print(f"第一个元素形状: {data[0].shape if hasattr(data[0], 'shape') else len(data[0])}")
