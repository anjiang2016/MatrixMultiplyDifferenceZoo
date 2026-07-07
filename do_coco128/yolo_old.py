"""
yolo_numpy.py - 纯 NumPy + funcs.py 实现的简化 YOLO
"""

import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import sys
# 将项目根目录添加到搜索路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# 导入 funcs.py 中的基础层
from funcs import conv, d_conv, relu, d_relu, sigmoid, d_sigmoid


# ============================================================
# 1. 模型构建
# ============================================================
def init_yolo_weights(num_classes=80):
    """
    初始化 YOLO 检测器权重
    """
    weights = {}
    
    def he_init(shape):
        if len(shape) == 4:
            fan_in = shape[1] * shape[2] * shape[3]
        else:
            fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape).astype(np.float32) * std
    
    # 骨干网络：4 个卷积块（stride=2 下采样）
    # conv1: 3 -> 16, 3x3, stride=2
    weights['conv1_w'] = he_init((16, 3, 3, 3))
    weights['conv1_b'] = np.zeros(16, dtype=np.float32)
    
    # conv2: 16 -> 32, 3x3, stride=2
    weights['conv2_w'] = he_init((32, 16, 3, 3))
    weights['conv2_b'] = np.zeros(32, dtype=np.float32)
    
    # conv3: 32 -> 64, 3x3, stride=2
    weights['conv3_w'] = he_init((64, 32, 3, 3))
    weights['conv3_b'] = np.zeros(64, dtype=np.float32)
    
    # conv4: 64 -> 128, 3x3, stride=2
    weights['conv4_w'] = he_init((128, 64, 3, 3))
    weights['conv4_b'] = np.zeros(128, dtype=np.float32)
    
    # conv5: 128 -> 256, 3x3, stride=2
    weights['conv5_w'] = he_init((256, 128, 3, 3))
    weights['conv5_b'] = np.zeros(256, dtype=np.float32)
    
    # 检测头：256 -> (5 + num_classes), 1x1 卷积
    head_channels = 5 + num_classes
    weights['head_w'] = he_init((head_channels, 256, 1, 1))
    weights['head_b'] = np.zeros(head_channels, dtype=np.float32)
    
    return weights


def forward_yolo(x, weights):
    """
    YOLO 前向传播
    x: (N, 3, 416, 416)
    weights: 模型权重字典
    返回: (N, 5+num_classes, 13, 13)
    """
    caches = {}
    
    # Conv1: stride=2, padding=1 (3x3 卷积，输出尺寸减半)
    out, x_pad, (H_out, W_out) = conv(x, weights['conv1_w'], weights['conv1_b'], stride=2, padding=1)
    caches['conv1'] = (x, weights['conv1_w'], weights['conv1_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu1'] = out
    
    # Conv2
    out, x_pad, (H_out, W_out) = conv(out, weights['conv2_w'], weights['conv2_b'], stride=2, padding=1)
    caches['conv2'] = (out, weights['conv2_w'], weights['conv2_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu2'] = out
    
    # Conv3
    out, x_pad, (H_out, W_out) = conv(out, weights['conv3_w'], weights['conv3_b'], stride=2, padding=1)
    caches['conv3'] = (out, weights['conv3_w'], weights['conv3_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu3'] = out
    
    # Conv4
    out, x_pad, (H_out, W_out) = conv(out, weights['conv4_w'], weights['conv4_b'], stride=2, padding=1)
    caches['conv4'] = (out, weights['conv4_w'], weights['conv4_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu4'] = out
    
    # Conv5
    out, x_pad, (H_out, W_out) = conv(out, weights['conv5_w'], weights['conv5_b'], stride=2, padding=1)
    caches['conv5'] = (out, weights['conv5_w'], weights['conv5_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu5'] = out
    
    # 检测头
    out, x_pad, (H_out, W_out) = conv(out, weights['head_w'], weights['head_b'], stride=1, padding=0)
    caches['head'] = (out, weights['head_w'], weights['head_b'], 1, 0, x_pad, H_out, W_out)
    # out shape: (N, 5+num_classes, 13, 13)
    
    return out, caches


# ============================================================
# 2. 损失函数
# ============================================================
def yolo_loss(pred, targets, num_classes, grid_size=13, lambda_coord=5.0, lambda_noobj=0.5):
    """
    简化 YOLO 损失（纯 NumPy）
    pred: (N, 5+num_classes, H, W) 模型输出
    targets: list of dict, 每个包含 'boxes' (Mx4), 'classes' (M,)
    """
    N, C, H, W = pred.shape
    assert H == grid_size and W == grid_size
    
    # 拆分预测
    # pred 的通道顺序: tx, ty, tw, th, obj, cls...
    pred_tx = pred[:, 0, :, :]   # (N, H, W)
    pred_ty = pred[:, 1, :, :]
    pred_tw = pred[:, 2, :, :]
    pred_th = pred[:, 3, :, :]
    pred_obj = pred[:, 4, :, :]
    pred_cls = pred[:, 5:, :, :]  # (N, num_classes, H, W)
    
    total_loss = 0
    
    for b in range(N):
        boxes = targets[b]['boxes']      # (M, 4) x_center, y_center, w, h (归一化)
        classes = targets[b]['classes']  # (M,)
        
        if len(boxes) == 0:
            continue
        
        # 构建目标张量
        # 初始化目标: tx, ty, tw, th, obj, cls_onehot
        tx_target = np.zeros((H, W), dtype=np.float32)
        ty_target = np.zeros((H, W), dtype=np.float32)
        tw_target = np.zeros((H, W), dtype=np.float32)
        th_target = np.zeros((H, W), dtype=np.float32)
        obj_target = np.zeros((H, W), dtype=np.float32)
        cls_target = np.zeros((num_classes, H, W), dtype=np.float32)
        
        # 对每个目标分配网格
        for m in range(len(boxes)):
            x_c, y_c, w, h = boxes[m]
            # 目标所在的网格索引
            gi = int(x_c * W)
            gj = int(y_c * H)
            gi = min(gi, W-1)
            gj = min(gj, H-1)
            
            # 计算目标偏移 (相对于网格左上角)
            tx_target[gj, gi] = x_c * W - gi
            ty_target[gj, gi] = y_c * H - gj
            tw_target[gj, gi] = np.log(w * W + 1e-6)
            th_target[gj, gi] = np.log(h * H + 1e-6)
            
            # 置信度目标为 1
            obj_target[gj, gi] = 1.0
            
            # 类别 one-hot
            cls = classes[m]
            cls_target[cls, gj, gi] = 1.0
        
        # 计算损失
        # 坐标损失（只计算有目标的网格）
        coord_mask = obj_target  # (H, W)
        loss_tx = np.sum(coord_mask * (pred_tx[b] - tx_target) ** 2)
        loss_ty = np.sum(coord_mask * (pred_ty[b] - ty_target) ** 2)
        loss_tw = np.sum(coord_mask * (pred_tw[b] - tw_target) ** 2)
        loss_th = np.sum(coord_mask * (pred_th[b] - th_target) ** 2)
        
        # 置信度损失（正样本 + 负样本）
        # 正样本置信度损失
        loss_obj_pos = np.sum(coord_mask * (pred_obj[b] - obj_target) ** 2)
        # 负样本置信度损失（忽略有目标的网格）
        noobj_mask = 1 - coord_mask
        loss_obj_noobj = np.sum(noobj_mask * (pred_obj[b] - 0) ** 2)
        
        # 类别损失（只计算有目标的网格）
        loss_cls = 0
        for c in range(num_classes):
            loss_cls += np.sum(coord_mask * (pred_cls[b, c] - cls_target[c]) ** 2)
        
        # 组合损失
        loss = (lambda_coord * (loss_tx + loss_ty + loss_tw + loss_th) +
                loss_obj_pos + lambda_noobj * loss_obj_noobj +
                loss_cls)
        total_loss += loss
    
    return total_loss / N


# ============================================================
# 3. 后处理（解码 + NMS）
# ============================================================
def decode_predictions(pred, img_size=416, conf_thresh=0.3, iou_thresh=0.5):
    """
    pred: (N, 5+num_classes, H, W)
    """
    N, C, H, W = pred.shape
    num_classes = C - 5
    
    detections = []
    
    for b in range(N):
        batch_dets = []
        for i in range(H):
            for j in range(W):
                tx = pred[b, 0, i, j]
                ty = pred[b, 1, i, j]
                tw = pred[b, 2, i, j]
                th = pred[b, 3, i, j]
                obj = sigmoid(pred[b, 4, i, j])
                cls = sigmoid(pred[b, 5:, i, j])
                
                # 置信度
                conf = obj * np.max(cls)
                if conf < conf_thresh:
                    continue
                
                cls_id = np.argmax(cls)
                
                # 解码边界框
                x_c = (j + tx) / W * img_size
                y_c = (i + ty) / H * img_size
                w = np.exp(tw) / W * img_size
                h = np.exp(th) / H * img_size
                
                x1 = x_c - w/2
                y1 = y_c - h/2
                x2 = x_c + w/2
                y2 = y_c + h/2
                
                batch_dets.append([x1, y1, x2, y2, conf, cls_id])
        
        if not batch_dets:
            detections.append([])
            continue
        
        # NMS
        batch_dets = np.array(batch_dets)
        keep = nms_numpy(batch_dets[:, :4], batch_dets[:, 4], iou_thresh)
        detections.append(batch_dets[keep].tolist())
    
    return detections


def nms_numpy(boxes, scores, iou_thresh=0.5):
    """NMS 纯 NumPy 实现"""
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    
    return keep


# ============================================================
# 4. 可视化工具
# ============================================================
def visualize_detections(image, dets, coco, title='Detections'):
    """显示图片和检测框"""
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
    
    # 类别名称
    cat_ids = coco.getCatIds()
    cat_names = [coco.loadCats([cid])[0]['name'] for cid in cat_ids]
    
    for det in dets:
        x1, y1, x2, y2, conf, cls_id = det
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-10, f'{cat_names[cls_id]}:{conf:.2f}',
                color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.6, edgecolor='none'))
    
    plt.show()



def load_yolo_dataset(data_dir='coco128', img_size=416, split='train2017'):
    """
    加载 YOLO 格式数据集
    
    Args:
        data_dir: 数据集根目录（如 coco128）
        img_size: 输入图片尺寸
        split: 子集名称（如 train2017）
    
    Returns:
        dataset: list of dict
        num_classes: 类别数（从 labels 中自动推断）
    """
    img_dir = os.path.join(data_dir, 'images', split)
    label_dir = os.path.join(data_dir, 'labels', split)
    
    # 获取所有图片文件
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    img_files.sort()
    
    dataset = []
    all_classes = set()
    
    print(f"加载 {len(img_files)} 张图片...")
    
    for img_file in img_files:
        # 图片路径
        img_path = os.path.join(img_dir, img_file)
        
        # 标签路径（同名 .txt）
        label_name = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        image = image.resize((img_size, img_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # (3, H, W)
        
        # 加载标签（如果存在）
        boxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append([x_c, y_c, w, h])
                    classes.append(cls_id)
                    all_classes.add(cls_id)
        
        dataset.append({
            'image': image,
            'boxes': np.array(boxes, dtype=np.float32),
            'classes': np.array(classes, dtype=np.int64),
            'img_path': img_path
        })
    
    num_classes = max(all_classes) + 1 if all_classes else 80
    print(f"完成加载，共 {len(dataset)} 张图片，{num_classes} 个类别")
    
    return dataset, num_classes
