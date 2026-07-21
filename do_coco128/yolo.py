"""
yolo_numpy.py - 纯 NumPy YOLO 检测器（基于 funcs.py）
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加父目录到路径，以便导入 funcs
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from funcs import conv, d_conv, relu, d_relu, sigmoid, d_sigmoid


# ============================================================
# 1. 模型初始化
# ============================================================
def init_yolo_weights(num_classes=80):
    """初始化 YOLO 检测器权重"""
    weights = {}

    def he_init(shape):
        if len(shape) == 4:
            fan_in = shape[1] * shape[2] * shape[3]
        else:
            fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape).astype(np.float32) * std

    # 骨干网络：5个卷积块，步长2下采样 (416->208->104->52->26->13)
    # conv1: 3->16, 3x3, stride=2, pad=1
    weights['conv1_w'] = he_init((16, 3, 3, 3))
    weights['conv1_b'] = np.zeros(16, dtype=np.float32)
    # conv2: 16->32, 3x3, stride=2, pad=1
    weights['conv2_w'] = he_init((32, 16, 3, 3))
    weights['conv2_b'] = np.zeros(32, dtype=np.float32)
    # conv3: 32->64, 3x3, stride=2, pad=1
    weights['conv3_w'] = he_init((64, 32, 3, 3))
    weights['conv3_b'] = np.zeros(64, dtype=np.float32)
    # conv4: 64->128, 3x3, stride=2, pad=1
    weights['conv4_w'] = he_init((128, 64, 3, 3))
    weights['conv4_b'] = np.zeros(128, dtype=np.float32)
    # conv5: 128->256, 3x3, stride=2, pad=1
    weights['conv5_w'] = he_init((256, 128, 3, 3))
    weights['conv5_b'] = np.zeros(256, dtype=np.float32)

    # 检测头：256 -> (5+num_classes), 1x1 conv
    head_channels = 5 + num_classes
    weights['head_w'] = he_init((head_channels, 256, 1, 1))
    weights['head_b'] = np.zeros(head_channels, dtype=np.float32)

    return weights


# ============================================================
# 2. 前向传播
# ============================================================
def forward_yolo(x, weights):
    caches = {}

    # Conv1
    conv1_input = x
    out, x_pad, (H_out, W_out) = conv(conv1_input, weights['conv1_w'], weights['conv1_b'], stride=2, padding=1)
    caches['conv1'] = (conv1_input, weights['conv1_w'], weights['conv1_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu1'] = out

    # Conv2
    conv2_input = out
    out, x_pad, (H_out, W_out) = conv(conv2_input, weights['conv2_w'], weights['conv2_b'], stride=2, padding=1)
    caches['conv2'] = (conv2_input, weights['conv2_w'], weights['conv2_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu2'] = out

    # Conv3
    conv3_input = out
    out, x_pad, (H_out, W_out) = conv(conv3_input, weights['conv3_w'], weights['conv3_b'], stride=2, padding=1)
    caches['conv3'] = (conv3_input, weights['conv3_w'], weights['conv3_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu3'] = out

    # Conv4
    conv4_input = out
    out, x_pad, (H_out, W_out) = conv(conv4_input, weights['conv4_w'], weights['conv4_b'], stride=2, padding=1)
    caches['conv4'] = (conv4_input, weights['conv4_w'], weights['conv4_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu4'] = out

    # Conv5
    conv5_input = out
    out, x_pad, (H_out, W_out) = conv(conv5_input, weights['conv5_w'], weights['conv5_b'], stride=2, padding=1)
    caches['conv5'] = (conv5_input, weights['conv5_w'], weights['conv5_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu5'] = out

    # Head
    head_input = out
    out, x_pad, (H_out, W_out) = conv(head_input, weights['head_w'], weights['head_b'], stride=1, padding=0)
    caches['head'] = (head_input, weights['head_w'], weights['head_b'], 1, 0, x_pad, H_out, W_out)

    return out, caches


# ============================================================
# 3. 损失函数
# ============================================================
def yolo_loss(pred, targets, num_classes, grid_size=13, lambda_coord=5.0, lambda_noobj=0.5,class_weights=None):
    """
    pred: (N, 5+num_classes, H, W)
    targets: list of dict {'boxes': (M,4), 'classes': (M,)}
    返回: scalar loss
    """
    N, C, H, W = pred.shape
    assert H == grid_size and W == grid_size

    pred_tx = pred[:, 0, :, :]
    pred_ty = pred[:, 1, :, :]
    pred_tw = pred[:, 2, :, :]
    pred_th = pred[:, 3, :, :]
    pred_obj = pred[:, 4, :, :]
    pred_cls = pred[:, 5:, :, :]  # (N, num_classes, H, W)

    total_loss = 0.0
    total_loss_box = 0.0          # 原始 box loss（未加权）
    total_loss_cls = 0.0          # 原始 cls loss（未加权）
    total_loss_obj = 0.0           # 原始 obj loss（未加权）
    for b in range(N):
        boxes = targets[b]['boxes']
        classes = targets[b]['classes']
        if len(boxes) == 0:
            continue

        tx_t = np.zeros((H, W), dtype=np.float32)
        ty_t = np.zeros((H, W), dtype=np.float32)
        tw_t = np.zeros((H, W), dtype=np.float32)
        th_t = np.zeros((H, W), dtype=np.float32)
        obj_t = np.zeros((H, W), dtype=np.float32)
        cls_t = np.zeros((num_classes, H, W), dtype=np.float32)

        for m in range(len(boxes)):
            x_c, y_c, w, h = boxes[m]
            gi = int(x_c * W)
            gj = int(y_c * H)
            gi = min(gi, W-1)
            gj = min(gj, H-1)

            tx_t[gj, gi] = x_c * W - gi
            ty_t[gj, gi] = y_c * H - gj
            tw_t[gj, gi] = np.log(w * W + 1e-6)
            th_t[gj, gi] = np.log(h * H + 1e-6)
            obj_t[gj, gi] = 1.0
            cls_id = classes[m]
            cls_t[cls_id, gj, gi] = 1.0

        coord_mask = obj_t  # (H, W)

        loss_tx = np.sum(coord_mask * (pred_tx[b] - tx_t) ** 2)
        loss_ty = np.sum(coord_mask * (pred_ty[b] - ty_t) ** 2)
        loss_tw = np.sum(coord_mask * (pred_tw[b] - tw_t) ** 2)
        loss_th = np.sum(coord_mask * (pred_th[b] - th_t) ** 2)

        loss_obj_pos = np.sum(coord_mask * (pred_obj[b] - obj_t) ** 2)
        noobj_mask = 1 - coord_mask
        loss_obj_noobj = np.sum(noobj_mask * (pred_obj[b] - 0) ** 2)

        loss_cls = 0.0
        for c in range(num_classes):
            if class_weights is not None:
                loss_cls += np.sum(coord_mask * class_weights[c]*(pred_cls[b, c] - cls_t[c]) ** 2)
            else:
                loss_cls += np.sum(coord_mask *(pred_cls[b, c] - cls_t[c]) ** 2)
                
        loss_box = 0.0
        loss_box = lambda_coord*(loss_tx+loss_ty+loss_tw+loss_th)
        loss_obj = 0.0
        loss_obj = 20.0*loss_obj_pos + 0.05 * loss_obj_noobj
        loss = loss_box + loss_obj + 6.0*loss_cls
        #loss = (lambda_coord * (loss_tx + loss_ty + loss_tw + loss_th) +
        #        loss_obj_pos + lambda_noobj * loss_obj_noobj +
        #        loss_cls)
        total_loss += loss
        total_loss_box +=20.0*loss_obj_pos
        total_loss_obj +=loss_obj
        total_loss_cls += 0.05*loss_obj_noobj
    components = {
        "box": total_loss_box/N,          # 原始 box loss（未加权）
        "cls": total_loss_cls/N,          # 原始 cls loss（未加权）
        "obj": total_loss_obj/N           # 原始 obj loss（未加权）
    }
    return total_loss/N , components


# ============================================================
# 4. 反向传播 (d_loss 对 pred 的梯度)
# ============================================================
def d_yolo_loss(pred, targets, num_classes, grid_size=13, lambda_coord=5.0, lambda_noobj=0.5):
    """
    计算损失对 pred 的梯度 (dL/dpred)
    pred: (N, 5+num_classes, H, W)
    返回: (N, 5+num_classes, H, W) 梯度
    """
    N, C, H, W = pred.shape
    d_pred = np.zeros_like(pred)

    pred_tx = pred[:, 0, :, :]
    pred_ty = pred[:, 1, :, :]
    pred_tw = pred[:, 2, :, :]
    pred_th = pred[:, 3, :, :]
    pred_obj = pred[:, 4, :, :]
    pred_cls = pred[:, 5:, :, :]

    for b in range(N):
        boxes = targets[b]['boxes']
        classes = targets[b]['classes']
        if len(boxes) == 0:
            continue

        tx_t = np.zeros((H, W), dtype=np.float32)
        ty_t = np.zeros((H, W), dtype=np.float32)
        tw_t = np.zeros((H, W), dtype=np.float32)
        th_t = np.zeros((H, W), dtype=np.float32)
        obj_t = np.zeros((H, W), dtype=np.float32)
        cls_t = np.zeros((num_classes, H, W), dtype=np.float32)

        for m in range(len(boxes)):
            x_c, y_c, w, h = boxes[m]
            gi = int(x_c * W)
            gj = int(y_c * H)
            gi = min(gi, W-1)
            gj = min(gj, H-1)
            tx_t[gj, gi] = x_c * W - gi
            ty_t[gj, gi] = y_c * H - gj
            tw_t[gj, gi] = np.log(w * W + 1e-6)
            th_t[gj, gi] = np.log(h * H + 1e-6)
            obj_t[gj, gi] = 1.0
            cls_t[classes[m], gj, gi] = 1.0

        coord_mask = obj_t

        # d_tx, d_ty, d_tw, d_th
        d_pred[b, 0, :, :] = 2 * lambda_coord * coord_mask * (pred_tx[b] - tx_t)
        d_pred[b, 1, :, :] = 2 * lambda_coord * coord_mask * (pred_ty[b] - ty_t)
        d_pred[b, 2, :, :] = 2 * lambda_coord * coord_mask * (pred_tw[b] - tw_t)
        d_pred[b, 3, :, :] = 2 * lambda_coord * coord_mask * (pred_th[b] - th_t)

        # d_obj
        d_pred[b, 4, :, :] = (2 * coord_mask * (pred_obj[b] - obj_t) +
                              2 * lambda_noobj * (1 - coord_mask) * pred_obj[b])

        # d_cls
        for c in range(num_classes):
            d_pred[b, 5+c, :, :] = 2 * coord_mask * (pred_cls[b, c] - cls_t[c])

    return d_pred / N


# ============================================================
# 5. 反向传播整个网络 (需要 d_conv, d_relu)
# ============================================================
def backward_yolo(dloss, caches):
    """
    dloss: 损失对输出的梯度 (N, 5+num_classes, 13, 13)
    caches: 前向传播缓存的各层信息
    返回: grads dict
    """
    grads = {}
    dx = dloss

    # Head 反向
    out, w, b, stride, padding, x_pad, H_out, W_out = caches['head']
    dx, dw, db = d_conv(dx, out, w, stride, padding, x_pad, H_out, W_out)
    grads['head_w'] = dw
    grads['head_b'] = db

    # ReLU5 反向
    dx = dx * d_relu(None, caches['relu5'])

    # Conv5 反向
    out, w, b, stride, padding, x_pad, H_out, W_out = caches['conv5']
    dx, dw, db = d_conv(dx, out, w, stride, padding, x_pad, H_out, W_out)
    grads['conv5_w'] = dw
    grads['conv5_b'] = db

    # ReLU4
    dx = dx * d_relu(None, caches['relu4'])
    # Conv4
    out, w, b, stride, padding, x_pad, H_out, W_out = caches['conv4']
    dx, dw, db = d_conv(dx, out, w, stride, padding, x_pad, H_out, W_out)
    grads['conv4_w'] = dw
    grads['conv4_b'] = db

    # ReLU3
    dx = dx * d_relu(None, caches['relu3'])
    # Conv3
    out, w, b, stride, padding, x_pad, H_out, W_out = caches['conv3']
    dx, dw, db = d_conv(dx, out, w, stride, padding, x_pad, H_out, W_out)
    grads['conv3_w'] = dw
    grads['conv3_b'] = db

    # ReLU2
    dx = dx * d_relu(None, caches['relu2'])
    # Conv2
    out, w, b, stride, padding, x_pad, H_out, W_out = caches['conv2']
    dx, dw, db = d_conv(dx, out, w, stride, padding, x_pad, H_out, W_out)
    grads['conv2_w'] = dw
    grads['conv2_b'] = db

    # ReLU1
    dx = dx * d_relu(None, caches['relu1'])
    # Conv1
    out, w, b, stride, padding, x_pad, H_out, W_out = caches['conv1']
    dx, dw, db = d_conv(dx, out, w, stride, padding, x_pad, H_out, W_out)
    grads['conv1_w'] = dw
    grads['conv1_b'] = db

    return grads


# ============================================================
# 6. 后处理（解码 + NMS）
# ============================================================
def decode_predictions(pred, img_size=416, conf_thresh=0.3, iou_thresh=0.5):
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
                #import pdb;pdb.set_trace()
                obj = sigmoid(pred[b, 4, i, j])
                cls = sigmoid(pred[b, 5:, i, j])
                conf = obj * np.max(cls)
                if conf < conf_thresh:
                    continue
                cls_id = np.argmax(cls)
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
        batch_dets = np.array(batch_dets)
        keep = nms_numpy(batch_dets[:, :4], batch_dets[:, 4], iou_thresh)
        detections.append(batch_dets[keep].tolist())
    return detections


def nms_numpy(boxes, scores, iou_thresh=0.5):
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
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


# ============================================================
# 7. 数据加载 (YOLO格式)
# ============================================================
def load_yolo_dataset(data_dir, img_size=416, split='train2017'):
    img_dir = os.path.join(data_dir, 'images', split)
    label_dir = os.path.join(data_dir, 'labels', split)
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_files.sort()

    dataset = []
    all_classes = set()
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size))
        img = np.array(img).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (3, H, W)

        boxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
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
            'image': img,
            'boxes': np.array(boxes, dtype=np.float32),
            'classes': np.array(classes, dtype=np.int64),
            'img_path': img_path
        })

    num_classes = max(all_classes) + 1 if all_classes else 80
    return dataset, num_classes


# ============================================================
# 8. 可视化
# ============================================================
def visualize_detections(image, dets, class_names=None, title='Detections'):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
    for det in dets:
        x1, y1, x2, y2, conf, cls_id = det
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        label = f'{class_names[cls_id] if class_names else cls_id}:{conf:.2f}'
        ax.text(x1, y1-10, label, color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.6, edgecolor='none'))
    plt.show()
