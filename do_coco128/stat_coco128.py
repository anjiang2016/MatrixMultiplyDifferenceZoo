"""
stat_coco128.py
统计 COCO128 数据集中每个类别的样本数量及占比
"""

import os
from collections import Counter

def stat_coco128(data_dir='/Users/zhaomingming/data_sets/coco128', split='train2017'):
    label_dir = os.path.join(data_dir, 'labels', split)
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    class_counter = Counter()
    total_objects = 0
    num_images = len(label_files)
    
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                class_counter[cls_id] += 1
                total_objects += 1
    
    # 按数量降序排列
    sorted_classes = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
    
    print(f"总图片数: {num_images}")
    print(f"总目标数: {total_objects}")
    print(f"平均每张图片目标数: {total_objects / num_images:.2f}\n")
    print("类别统计 (类别ID: 样本数, 占比):")
    for cls_id, count in sorted_classes:
        ratio = count / total_objects * 100
        print(f"  {cls_id:2d}: {count:4d} ({ratio:5.2f}%)")
    
    return class_counter

if __name__ == "__main__":
    stat_coco128()
