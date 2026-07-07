"""
train_yolo.py - 训练 YOLO 检测器（纯 NumPy）
"""

import numpy as np
import os
import time
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import sys
from yolo import (
    init_yolo_weights, forward_yolo, yolo_loss,
    decode_predictions, visualize_detections
)
# 将项目根目录添加到搜索路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from funcs import d_conv, d_relu, d_sigmoid  # 用于反向传播


def load_coco128(data_dir='/Users/zhaomingming/data_sets/coco128', img_size=416):
    """加载 COCO128 数据集"""
    ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    num_classes = len(cat_ids)
    
    print(f"加载 {len(img_ids)} 张图片, {num_classes} 个类别")
    
    dataset = []
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(data_dir, 'images', 'train2017', img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        image = image.resize((img_size, img_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # (3, H, W)
        
        # 获取标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        boxes = []
        classes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x_c = (x + w/2) / orig_w
            y_c = (y + h/2) / orig_h
            w_n = w / orig_w
            h_n = h / orig_h
            boxes.append([x_c, y_c, w_n, h_n])
            classes.append(cat_id_to_idx[ann['category_id']])
        
        dataset.append({
            'image': image,
            'boxes': np.array(boxes, dtype=np.float32),
            'classes': np.array(classes, dtype=np.int64),
            'img_id': img_id
        })
    
    return dataset, num_classes


def backward_yolo(dloss, caches):
    """
    反向传播 YOLO 模型（需要实现每个层的反向）
    这里省略，因为训练纯 NumPy 太慢，仅做演示
    """
    # 实际需要实现 d_conv 和 d_relu 的反向传播
    # 参考 funcs.py 中的 d_conv 和 d_relu
    pass


def train_yolo_numpy(dataset, weights, num_classes, epochs=10, batch_size=4, lr=0.001):
    """训练 YOLO（纯 NumPy）"""
    N = len(dataset)
    num_batches = int(np.ceil(N / batch_size))
    
    print(f"训练 {N} 张图片, {num_batches} 个 batch")
    print("-" * 60)
    
    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(N)
        total_loss = 0
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, N)
            batch_indices = indices[start:end]
            
            # 构建 batch
            batch_images = np.stack([dataset[i]['image'] for i in batch_indices], axis=0)
            batch_targets = [{'boxes': dataset[i]['boxes'], 'classes': dataset[i]['classes']} 
                             for i in batch_indices]
            
            # 前向
            pred, caches = forward_yolo(batch_images, weights)
            
            # 计算损失
            loss = yolo_loss(pred, batch_targets, num_classes)
            total_loss += loss
            
            # 反向（这里省略，需要实现 d_conv, d_relu 的反向传播）
            # 由于纯 NumPy 训练极慢，此部分仅示意
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        print("-" * 40)
    
    return weights


def main():
    # 配置
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'
    IMG_SIZE = 416
    EPOCHS = 1  # 仅演示，实际需要更多
    BATCH_SIZE = 2
    LR = 0.001
    
    # 下载数据集（如果不存在）
    if not os.path.exists(DATA_DIR):
        print("下载 COCO128...")
        import requests, zipfile
        url = "https://ultralytics.com/assets/coco128.zip"
        r = requests.get(url)
        with open('coco128.zip', 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile('coco128.zip', 'r') as z:
            z.extractall('.')
        os.remove('coco128.zip')
    
    # 加载数据
    dataset, num_classes = load_coco128(DATA_DIR, IMG_SIZE)
    
    # 初始化权重
    weights = init_yolo_weights(num_classes)
    
    # 训练（纯 NumPy 非常慢，建议只跑少量 epoch 演示）
    print("警告：纯 NumPy 训练检测模型非常慢，建议减少 epoch 或使用 PyTorch")
    weights = train_yolo_numpy(dataset, weights, num_classes, epochs=EPOCHS, 
                               batch_size=BATCH_SIZE, lr=LR)
    
    # 保存权重
    np.savez('yolo_weights.npz', **weights)
    print("权重已保存到 yolo_weights.npz")
    
    # 测试单张图片推理
    sample = dataset[0]
    image = sample['image'][np.newaxis, ...]
    pred, _ = forward_yolo(image, weights)
    dets = decode_predictions(pred, img_size=IMG_SIZE, conf_thresh=0.3)
    
    # 可视化
    coco = COCO(os.path.join(DATA_DIR, 'annotations', 'instances_train2017.json'))
    from yolo_numpy import visualize_detections
    img_display = (sample['image'].transpose(1, 2, 0) * 255).astype(np.uint8)
    visualize_detections(img_display, dets[0], coco, title='YOLO Detections')


if __name__ == "__main__":
    main()
