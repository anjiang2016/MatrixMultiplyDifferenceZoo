## 主要代码作用
```
训练yoloreadme.md
.主要说经train_yolo_html1.py是训练主代码。
evaluate.py是计算mAp的，
infer_manual.py是查看单张结果的。
server.py启动网页服务器，
http://localhost:8000/?start_points=0产看loss曲线
dsp_cooc128.py 查看数据集
```

## YOLO 纯 NumPy 实现 - 基于 MMD 矩阵求导库
```
不用 PyTorch，不用 TensorFlow，纯 NumPy + 手写反向传播实现 YOLO 目标检测
```
### 📖 项目简介
```
本项目完全从零实现了 YOLO 目标检测器，核心亮点是纯 NumPy 实现 + 手写反向传播，不依赖任何深度学习框架。项目包含完整的训练、评估、推理和可视化监控工具链。
```
#### 核心设计理念：

- 所有前向/反向传播手写实现，梯度流动完全透明

- 基于 MMD（MatrixMultiplyDifferentiationZoo）矩阵求导库

- 支持类别加权损失，有效处理数据不平衡

- 实时 Web 监控训练过程

### 🚀 快速开始
#### 环境依赖
```
bash
pip install numpy matplotlib pillow
```
#### 数据准备
COCO128 数据集（128 张图片，80 个类别）会自动从 data_sets/coco128 加载。

#### 训练模型
```
bash
python train_yolo_html1.py
```

训练过程中会自动：

- 保存最佳模型到 best_model.npz

- 每轮 epoch 记录日志到 training_log_class_weight.json

- 监控页面实时显示 Loss 曲线

### 📁 核心脚本说明
```
脚本	功能	使用方式
train_yolo_html1.py	训练主脚本	python train_yolo_html1.py
evaluate.py	计算 mAP 和各类别 AP	python evaluate.py
infer_manual.py	手动浏览单张图片推理结果	python infer_manual.py
server.py	启动 Web 监控服务器	python server.py
```
### 训练监控

启动 Web 服务器查看实时训练曲线：
```
bash
python server.py
```
然后打开浏览器访问：
```
text
http://localhost:8000/?start_points=0
```
参数说明：

- start_points=N：从第 N 个 epoch 开始显示（用于放大查看后期曲线）

- 不加参数则显示全部数据

### 推理可视化
```
bash
python infer_manual.py
```
#### 操作方式：

- 左右方向键切换图片

- q 键退出

### 📊 训练效果（COCO128）
| 指标|	值|
|---|---|
|训练集| mAP@0.5待评估|
|最佳 Loss|	2.58（epoch 600）|
|数据量|128 张图片，80 类|
#### 训练 Loss 下降曲线
Loss 从 643 稳步下降到 2.58，比例法类别权重策略有效收敛。
```
text
Epoch 1:  643.7
Epoch 100: 58.8
Epoch 300: 15.0
Epoch 500: 4.8
Epoch 600: 2.58
```
#### ⚙️ 关键训练策略
##### 类别权重策略
采用逆频率比例法解决 COCO128 类别严重不平衡问题（person 占 27%，大量类别仅 1-2 个样本）：
```
python
权重 = total_samples / class_count
# 归一化后最大值拉伸到 1.0
```
##### 损失函数
YOLO 损失 = 坐标损失 + 置信度损失 + 分类损失：

- loss_box：边界框回归（MSE）

- loss_obj：目标置信度（BCE）

- loss_cls：类别概率（BCE）

##### 优化器
- Adam 优化器（纯 NumPy 实现）

- 阶梯学习率衰减（每 20 epoch × 0.9）

### 🧪 评估模型
```
bash
python evaluate.py
```
#### 输出内容：

- 每个类别的 AP（Average Precision）

- 每个类别的最佳 F1-score 及对应的置信度阈值

- 整体 mAP@0.5

### 📂 目录结构
```
text
do_coco128/
├── train_yolo_html1.py      # 训练主脚本
├── evaluate.py              # mAP 评估
├── infer_manual.py          # 推理可视化
├── server.py                # Web 监控服务器
├── yolo.py                  # 模型定义（前向/反向/损失）
├── funcs.py                 # 基础层（conv, relu, adam, ...）
├── best_model.npz           # 最佳模型权重
├── training_log_class_weight.json  # 训练日志
└── models/                  # 模型检查点目录
```
### 💡 经验总结
- 类别权重策略至关重要：逆频率比例法比复杂的 gamma 调度更简单有效

- 纯 NumPy 训练慢但透明：适合教学和理解反向传播细节

- Web 监控实时反馈：训练过程中即可观察 Loss 趋势，及时调整

- 置信度阈值调优：通过 F1-score 选择最佳阈值，平衡精确率和召回率

### 📌 后续计划
- 支持完整 COCO 数据集训练

- C 扩展加速卷积运算

### 📄 License
MIT
