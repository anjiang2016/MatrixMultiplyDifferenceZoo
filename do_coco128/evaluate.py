"""
evaluate.py
在训练集上计算 mAP、各类别 AP 以及最佳 F1-score（含最佳置信度阈值）
"""

import os
import sys
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from yolo import (
    forward_yolo, decode_predictions,
    load_yolo_dataset
)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def load_weights(best_model_path='best_model.npz'):
    """加载最佳模型权重"""
    best_data = np.load(best_model_path, allow_pickle=True)
    weights = {k: best_data[k] for k in best_data.files if k not in ['best_loss', 'best_epoch']}
    return weights


def get_gt_boxes(sample, img_size=416):
    """从样本中提取真实框 (x1,y1,x2,y2,cls) 像素坐标"""
    boxes = sample['boxes']
    classes = sample['classes']
    gt = []
    for i in range(len(boxes)):
        cx, cy, w, h = boxes[i]
        x1 = (cx - w/2) * img_size
        y1 = (cy - h/2) * img_size
        x2 = (cx + w/2) * img_size
        y2 = (cy + h/2) * img_size
        gt.append([x1, y1, x2, y2, int(classes[i])])
    return np.array(gt)


def compute_iou(box1, box2):
    """计算两个框的 IoU (box: x1,y1,x2,y2)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


def evaluate_dataset(dataset, weights, img_size=416, conf_thresh=0.01, iou_thresh=0.5):
    """
    对整个数据集进行推理，返回每个类别的 TP/FP 和置信度列表
    """
    all_detections = defaultdict(list)  # cls_id: list of (conf, tp)
    all_gt_count = defaultdict(int)     # cls_id: total gt count

    for idx, sample in enumerate(dataset):
        print(f"处理图片 {idx+1}/{len(dataset)}", end='\r')
        img = sample['image'][np.newaxis, ...]
        gt_boxes = get_gt_boxes(sample, img_size)
        for box in gt_boxes:
            cls = int(box[4])
            all_gt_count[cls] += 1

        pred, _ = forward_yolo(img, weights)
        dets = decode_predictions(pred, img_size=img_size, conf_thresh=conf_thresh)
        pred_boxes = dets[0]

        matched = [False] * len(gt_boxes)
        for pred_box in pred_boxes:
            x1, y1, x2, y2, conf, cls = pred_box
            cls = int(cls)
            best_iou = iou_thresh
            best_idx = -1
            for i, gt in enumerate(gt_boxes):
                if gt[4] == cls:
                    iou = compute_iou([x1,y1,x2,y2], gt[:4])
                    if iou > best_iou and not matched[i]:
                        best_iou = iou
                        best_idx = i
            if best_idx >= 0:
                matched[best_idx] = True
                all_detections[cls].append((conf, 1))
            else:
                all_detections[cls].append((conf, 0))

    print("\n评估完成。")
    return all_detections, all_gt_count


def compute_ap(recall, precision):
    """计算 AP (11-point interpolation)"""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap


def compute_map(all_detections, all_gt_count):
    """计算每个类别的 AP 和 mAP"""
    ap_per_class = {}
    for cls, dets in all_detections.items():
        if cls not in all_gt_count or all_gt_count[cls] == 0:
            ap_per_class[cls] = 0.0
            continue
        dets_sorted = sorted(dets, key=lambda x: x[0], reverse=True)
        tp = np.array([d[1] for d in dets_sorted])
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1 - tp)
        n_gt = all_gt_count[cls]
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall = tp_cum / n_gt
        ap = compute_ap(recall, precision)
        ap_per_class[cls] = ap
    return ap_per_class


def compute_best_f1(all_detections, all_gt_count, num_thresholds=100):
    """
    计算每个类别的最佳 F1-score 及对应的置信度阈值
    返回: dict {cls: (best_f1, best_conf)}
    """
    best_f1_per_class = {}
    for cls, dets in all_detections.items():
        if cls not in all_gt_count or all_gt_count[cls] == 0:
            best_f1_per_class[cls] = (0.0, 0.0)
            continue
        # 按置信度降序排列
        dets_sorted = sorted(dets, key=lambda x: x[0], reverse=True)
        confs = np.array([d[0] for d in dets_sorted])
        tp = np.array([d[1] for d in dets_sorted])
        n_gt = all_gt_count[cls]
        # 如果没有任何预测框，F1=0
        if len(confs) == 0:
            best_f1_per_class[cls] = (0.0, 0.0)
            continue

        # 遍历可能的置信度阈值（从最高到最低，或者从0到1均匀采样）
        # 更精确：以每个预测框的置信度作为候选阈值
        best_f1 = 0.0
        best_conf = 0.0
        # 增加一个极小阈值确保能够考虑所有框
        thresholds = np.unique(confs)
        if len(thresholds) == 0:
            best_f1_per_class[cls] = (0.0, 0.0)
            continue
        # 从高到低尝试，也可以添加0.0作为最低阈值
        thresholds = np.sort(thresholds)[::-1]
        # 也可以加入0.0作为阈值
        thresholds = np.append(thresholds, 0.0)
        for t in thresholds:
            # 选择置信度 >= t 的框作为预测
            mask = confs >= t
            if np.sum(mask) == 0:
                continue
            tp_selected = tp[mask]
            fp_selected = 1 - tp_selected
            tp_sum = np.sum(tp_selected)
            fp_sum = np.sum(fp_selected)
            precision = tp_sum / (tp_sum + fp_sum + 1e-8)
            recall = tp_sum / (n_gt + 1e-8)
            if precision + recall == 0:
                continue
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_conf = t
        best_f1_per_class[cls] = (best_f1, best_conf)
    return best_f1_per_class


def main():
    DATA_DIR = '/Users/zhaomingming/data_sets/coco128'
    IMG_SIZE = 416
    CONF_THRESH = 0.01  # 评估时用低阈值，确保不遗漏
    IOU_THRESH = 0.5

    print("加载模型...")
    weights = load_weights('best_model.npz')
    print("加载数据集...")
    dataset, _ = load_yolo_dataset(DATA_DIR, IMG_SIZE)
    print(f"数据集大小: {len(dataset)} 张图片")

    print("开始评估...")
    all_detections, all_gt_count = evaluate_dataset(
        dataset, weights, IMG_SIZE, CONF_THRESH, IOU_THRESH
    )

    print("\n计算 AP...")
    ap_per_class = compute_map(all_detections, all_gt_count)

    print("\n计算最佳 F1-score...")
    best_f1_per_class = compute_best_f1(all_detections, all_gt_count)

    # 输出结果
    print("\n========== 各类别 AP 与最佳 F1 ==========")
    ap_list = []
    f1_list = []
    for cls in sorted(ap_per_class.keys()):
        cls_name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"cls{cls}"
        ap = ap_per_class[cls]
        ap_list.append(ap)
        f1, conf = best_f1_per_class.get(cls, (0.0, 0.0))
        f1_list.append(f1)
        print(f"{cls_name:20s} AP = {ap:.4f}  最佳 F1 = {f1:.4f} @ 置信度 = {conf:.4f}")

    mAP = np.mean(ap_list) if ap_list else 0.0
    avg_f1 = np.mean(f1_list) if f1_list else 0.0
    print(f"\nmAP@0.5 = {mAP:.4f}")
    print(f"平均最佳 F1 = {avg_f1:.4f}")
    conf_list = [conf for _, conf in best_f1_per_class.values() if conf > 0]
    avg_conf = np.mean(conf_list) if conf_list else 0.0
    print(f"平均最佳置信度 = {avg_conf:.4f}")

if __name__ == "__main__":
    main()
