# MNIST 手写数字分类 — 从零实现 LeNet-5

## 纯 NumPy 实现 LeNet-5，在 MNIST 上达到 98%+ 准确率
### 📖 项目简介

本项目从零实现了一个经典的 LeNet-5 卷积神经网络，用于 MNIST 手写数字分类。整个项目不依赖 PyTorch、TensorFlow 等任何深度学习框架，所有组件（卷积、池化、全连接、反向传播、优化器）均使用 NumPy 手写实现。

### 核心亮点：

✅ 纯 NumPy 实现，无任何深度学习框架依赖
✅ 完整的前向传播 + 反向传播（手写梯度）
✅ 支持 断点续训，随时保存和加载模型
✅ LeNet-5 经典架构（2 个卷积层 + 2 个池化层 + 3 个全连接层）
✅ 训练稳定，测试准确率 98%+
### 🎯 项目意义
```
理解深度学习的“学习”本质：

深度学习就是一堆矩阵乘法
反向传播就是链式法则求导
训练过程就是梯度下降优化
通过从零实现，你可以亲眼看到每一层的梯度如何流动，权重如何更新，真正理解神经网络的工作原理。
```
### 🧠 模型架构

```
LeNet-5 结构

层	操作	输入 → 输出	参数
Input	-	32×32×1	-
Conv1	5×5, stride=1, pad=0	32×32×1 → 28×28×6	156
Sigmoid1	激活	28×28×6	-
Pool1	2×2, stride=2	28×28×6 → 14×14×6	-
Conv2	5×5, stride=1, pad=0	14×14×6 → 10×10×16	2,416
Sigmoid2	激活	10×10×16	-
Pool2	2×2, stride=2	10×10×16 → 5×5×16	-
Conv3	5×5, stride=1, pad=0	5×5×16 → 1×1×120	48,120
Sigmoid3	激活	1×1×120	-
Flatten	展平	1×1×120 → 120	-
FC1	全连接	120 → 84	10,164
Sigmoid4	激活	84	-
FC2	全连接	84 → 10	850
```
### 尺寸流动图

text```
Input: (1, 32, 32)
   ↓ Conv1 (5×5, pad=0)
(6, 28, 28)
   ↓ Pool1 (2×2, stride=2)
(6, 14, 14)
   ↓ Conv2 (5×5, pad=0)
(16, 10, 10)
   ↓ Pool2 (2×2, stride=2)
(16, 5, 5)
   ↓ Conv3 (5×5, pad=0)
(120, 1, 1)
   ↓ Flatten
(120,)
   ↓ FC1
(84,)
   ↓ FC2
(10,)  ← 输出 (10 个类别的 logits)
```

## 🚀 快速开始

#### 环境要求
```
bash
Python 3.6+
numpy
matplotlib
安装依赖

bash
pip install numpy matplotlib
```
### 数据准备
```
将 MNIST 数据保存为 mnist.pkl 格式，包含：

(X_train, y_train)：训练集
(X_val, y_val)：验证集
(X_test, y_test)：测试集
```
### 训练模型
```
bash
python train.py
```

### 断点续训
```
python
# 从第 10 个 epoch 的模型继续训练
weights, _, _, _, _ = train_mnist(
    pkl_path='mnist.pkl',
    resume_from='models/model_epoch_010.npz',
    epochs=20,
    learning_rate=0.01
)
```

### 📊 训练结果

### 调参历程
```
版本	激活函数	初始化	学习率	准确率	说明
v1	Sigmoid	Xavier	0.01	~10%	梯度消失，无法收敛
v2	Sigmoid	Xavier	0.001	~10%	梯度消失，无法收敛
v3	ReLU	He	0.01	98%+	✅ 成功收敛
```

## 训练曲线（5000 张）
```
text
Epoch 1/20 - Loss: 1.3953, Train Acc: 0.5531, Val Acc: 0.8230
Epoch 5/20 - Loss: 0.1927, Train Acc: 0.9427, Val Acc: 0.9260
Epoch 10/20 - Loss: 0.0936, Train Acc: 0.9728, Val Acc: 0.9500
Epoch 14/20 - Loss: 0.0621, Train Acc: 0.9818, Val Acc: 0.9600
最终测试集准确率：98%+
```
###📁 文件结构
```
text
project/
├── funcs.py              # 基础层（conv, relu, avgpool, linear, softmax, ...）
├── train.py              # 训练主程序
├── models/               # 保存的模型文件
│   ├── model_best.npz           # 最佳模型
│   ├── model_epoch_xxx.npz      # 检查点
│   └── ...
└── mnist.pkl             # MNIST 数据文件
```
### 文件说明
```
文件	职责
funcs.py	所有基础层：卷积、池化、全连接、激活函数及其导数（纯 NumPy）
train.py	训练循环：数据加载、前向传播、反向传播、参数更新、验证、保存
```
### 🛠️ 核心实现

1. 卷积层（前向 + 反向）
```
python
def conv(x, w, b=None, stride=1, padding=0):
    # 使用 im2col + 矩阵乘法加速
    # ...
    return out, x_pad, (H_out, W_out)

def d_conv(dout, x, w, stride=1, padding=0):
    # 反向传播：计算 dx, dw, db
    # ...
    return dx, dw, db
```	
2. ReLU 激活函数
```
python
def relu(x):
    return np.maximum(0, x)

def d_relu(x, y=None):
    if y is not None:
        return (y > 0).astype(np.float32)
    return (x > 0).astype(np.float32)
```		
3. 训练循环
```
python
def train_step(x, y, weights, learning_rate):
    # 前向
    logits, caches = forward_pass(x, weights)
    
    # 损失
    loss, ce_cache = softmax_cross_entropy(logits, y)
    
    # 反向
    dlogits = d_softmax_cross_entropy(ce_cache)
    grads = backward_pass(dlogits, caches, weights)
    
    # 更新
    for key in weights.keys():
        weights[key] -= learning_rate * grads[key]
    
    return loss, acc
```	
### 💡 经验总结

#### 为什么训练不收敛？
```
问题	症状	解决方案
梯度消失	Loss 卡在 2.3，准确率 ~10%	Sigmoid → ReLU
权重初始化不当	Loss 不下降	Xavier → He 初始化
学习率太小	收敛极慢	从 0.01 开始尝试
数据未归一化	训练不稳定	像素值归一化到 [0,1]
```
#### 关键调参经验
```
激活函数选 ReLU：深层网络避免梯度消失
初始化用 He：适配 ReLU
学习率从 0.01 开始：观察 Loss 下降速度
数据先归一化：让梯度更稳定
先小数据调试：100 张图快速验证代码正确性
用断点调试：比 print 大法快 10 倍
```
#### 📈 训练建议

小规模测试（快速验证）
```
python
num_samples = 100
epochs = 10
learning_rate = 0.1
正式训练

python
num_samples = 50000   # 全量数据
epochs = 20
learning_rate = 0.01
断点续训

python
resume_from = 'models/model_epoch_010.npz'
```
### 🔧 模型保存与加载
```
python
# 保存
save_model(weights, 'models/my_model.npz', epoch=10, loss=0.5, acc=0.98)

# 加载
weights, metadata = load_model('models/my_model.npz')
print(metadata)  # {'epoch': 10, 'loss': 0.5, 'acc': 0.98}

# 列出所有模型
list_models('models')
```
## 🎯 下一步
```
添加 Dropout 防止过拟合
实现 Batch Normalization
尝试 Adam 优化器
可视化卷积核和特征图
迁移到 CIFAR-10 数据集
```
### 📄 License

MIT

### 从零写 AI，理解深度学习的本质。🚀

