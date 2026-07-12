"""
create_coco128_32x32.py
将 COCO128 图片缩放到 32x32，并过滤掉缩放后宽或高 <=4 像素的标注框
"""

import os
import shutil
from PIL import Image
import numpy as np


def convert_yolo_to_pixel(x_c, y_c, w, h, img_size):
    """将归一化 YOLO 坐标转换为像素坐标 (x1,y1,x2,y2)"""
    x1 = (x_c - w / 2) * img_size
    y1 = (y_c - h / 2) * img_size
    x2 = (x_c + w / 2) * img_size
    y2 = (y_c + h / 2) * img_size
    return x1, y1, x2, y2


def convert_pixel_to_yolo(x1, y1, x2, y2, img_size):
    """将像素坐标 (x1,y1,x2,y2) 转换为归一化 YOLO 坐标"""
    w = x2 - x1
    h = y2 - y1
    x_c = (x1 + x2) / 2 / img_size
    y_c = (y1 + y2) / 2 / img_size
    w = w / img_size
    h = h / img_size
    return x_c, y_c, w, h


def process_dataset(src_dir, dst_dir, target_size=32, min_pixel_size=4):
    """
    处理整个数据集
    src_dir: 源数据集根目录（包含 images/ 和 labels/）
    dst_dir: 目标数据集根目录
    target_size: 缩放目标尺寸（正方形）
    min_pixel_size: 最小像素尺寸阈值，宽或高 <= 此值将被丢弃
    """
    split = 'train2017'
    src_img_dir = os.path.join(src_dir, 'images', split)
    src_lbl_dir = os.path.join(src_dir, 'labels', split)
    
    dst_img_dir = os.path.join(dst_dir, 'images', split)
    dst_lbl_dir = os.path.join(dst_dir, 'labels', split)
    
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    
    # 获取所有图片文件
    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_files.sort()
    
    total_boxes = 0
    kept_boxes = 0
    discarded_boxes = 0
    
    for img_file in img_files:
        # 图片路径
        src_img_path = os.path.join(src_img_dir, img_file)
        # 对应的标注文件
        label_name = os.path.splitext(img_file)[0] + '.txt'
        src_lbl_path = os.path.join(src_lbl_dir, label_name)
        
        # 读取图片并缩放
        img = Image.open(src_img_path).convert('RGB')
        # 兼容不同 Pillow 版本的重采样参数
        try:
            img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        except AttributeError:
            try:
                img_resized = img.resize((target_size, target_size), Image.LANCZOS)
            except AttributeError:
                img_resized = img.resize((target_size, target_size), Image.ANTIALIAS)
        
        # 读取标注
        boxes = []
        if os.path.exists(src_lbl_path):
            with open(src_lbl_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                # 转换为像素坐标
                x1, y1, x2, y2 = convert_yolo_to_pixel(x_c, y_c, w, h, target_size)
                # 裁剪到图像边界
                x1 = max(0, min(target_size, x1))
                y1 = max(0, min(target_size, y1))
                x2 = max(0, min(target_size, x2))
                y2 = max(0, min(target_size, y2))
                # 计算新宽度和高度
                new_w = x2 - x1
                new_h = y2 - y1
                # 如果宽度或高度 <= 阈值，丢弃该框
                if new_w <= min_pixel_size or new_h <= min_pixel_size:
                    discarded_boxes += 1
                    continue
                # 保留该框，转换回归一化坐标
                x_c_new, y_c_new, w_new, h_new = convert_pixel_to_yolo(x1, y1, x2, y2, target_size)
                # 确保归一化坐标在 [0,1] 范围内
                x_c_new = max(0, min(1, x_c_new))
                y_c_new = max(0, min(1, y_c_new))
                w_new = max(0, min(1, w_new))
                h_new = max(0, min(1, h_new))
                # 如果宽或高变为 0（由于浮点误差），跳过
                if w_new == 0 or h_new == 0:
                    discarded_boxes += 1
                    continue
                boxes.append((cls_id, x_c_new, y_c_new, w_new, h_new))
                kept_boxes += 1
            total_boxes += len(boxes)  # 仅统计原标注中的框数，但我们需要在读取时统计
        else:
            # 没有标注文件，跳过此图片？（通常都有，但以防万一）
            continue
        
        # 保存缩放后的图片
        dst_img_path = os.path.join(dst_img_dir, img_file)
        # 如果图片格式不是 JPEG，我们可以统一保存为 JPEG，但为了简单，保留原扩展名
        # 但原扩展名可能不是 .jpg，我们保持原样
        img_resized.save(dst_img_path)
        
        # 保存标注（即使没有框，也创建空文件）
        dst_lbl_path = os.path.join(dst_lbl_dir, label_name)
        with open(dst_lbl_path, 'w') as f:
            for cls_id, x_c, y_c, w, h in boxes:
                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        
        # 更新统计（这里重新统计实际保留的框数）
        # 但由于我们已有 kept_boxes 和 discarded_boxes，可在循环内累加
    
    print(f"处理完成！")
    print(f"总原始标注框数: {total_boxes}")
    print(f"保留的框数: {kept_boxes}")
    print(f"丢弃的框数: {discarded_boxes}")
    print(f"目标数据集保存在: {dst_dir}")


if __name__ == '__main__':
    SOURCE_DIR = '/Users/zhaomingming/data_sets/coco128'  # 修改为你的 COCO128 路径
    TARGET_DIR = './coco128_32x32'  # 输出到当前目录下的 coco128_32x32
    TARGET_SIZE = 32
    MIN_PIXEL_SIZE = 4  # 丢弃宽或高 <= 4 像素的框
    
    process_dataset(SOURCE_DIR, TARGET_DIR, TARGET_SIZE, MIN_PIXEL_SIZE)
