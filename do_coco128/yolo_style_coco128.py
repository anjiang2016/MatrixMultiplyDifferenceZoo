import cv2
import numpy as np
from pathlib import Path

# ==== 请根据你的实际路径修改这里 ====
img_dir = Path("/Users/zhaomingming/data_sets/coco128/images/train2017")
label_dir = Path("/Users/zhaomingming/data_sets/coco128/labels/train2017")

# COCO 128 的类别名称（共80个，这里只截取前几个示例，完整列表网上可查）
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

def draw_yolo_boxes(img_path, label_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取图片: {img_path}")
        return
    h, w = img.shape[:2]

    if not label_path.exists():
        print(f"标签文件不存在: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_c, y_c, box_w, box_h = map(float, parts)

        # 将归一化坐标转换为像素坐标（左上角和右下角）
        x1 = int((x_c - box_w / 2) * w)
        y1 = int((y_c - box_h / 2) * h)
        x2 = int((x_c + box_w / 2) * w)
        y2 = int((y_c + box_h / 2) * h)

        # 边界保护（防止画到图像外面）
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 获取类别名称（如果超出范围则显示 ID）
        class_name = COCO_CLASSES[int(class_id)] if int(class_id) < len(COCO_CLASSES) else str(int(class_id))

        # 画框和标签
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 显示
    cv2.imshow('YOLO Annotation', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==== 测试读取第一张图 ====
if __name__ == "__main__":
    # 获取第一张 jpg 图片
    first_img = list(img_dir.glob("*.jpg"))[0]
    corresponding_label = label_dir / (first_img.stem + ".txt")
    print(f"读取图片: {first_img.name}")
    draw_yolo_boxes(first_img, corresponding_label)
