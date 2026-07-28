# view_nuscenes_samples.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# ========== 配置 ==========
DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"   # 你的路径
ORDER = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
         'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']

# ========== 工具函数 ==========
def load_json(rel_path):
    with open(os.path.join(DATA_ROOT, rel_path), 'r') as f:
        return json.load(f)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def transform_points(points, rot, trans):
    return points @ rot.T + trans

def project_3d_box_to_2d(box_center, box_size, box_quat, cam_intrin, cam_extrin, ego_pose):
    """
    将 3D 框投影到 2D 图像平面（返回 2D 包围盒 [x1, y1, x2, y2]）
    """
    ego_pose_inv = np.linalg.inv(ego_pose)
    center_ego = transform_points(np.array(box_center), ego_pose_inv[:3, :3], ego_pose_inv[:3, 3])
    cam_extrin_inv = np.linalg.inv(cam_extrin)
    center_cam = transform_points(center_ego, cam_extrin_inv[:3, :3], cam_extrin_inv[:3, 3])
    if center_cam[2] <= 0:
        return None

    w, l, h = box_size
    hw, hl, hh = w/2, l/2, h/2
    corners_local = np.array([
        [ hl,  hw,  hh], [ hl, -hw,  hh], [ hl,  hw, -hh], [ hl, -hw, -hh],
        [-hl,  hw,  hh], [-hl, -hw,  hh], [-hl,  hw, -hh], [-hl, -hw, -hh]
    ])
    rot_global = quaternion_to_rotation_matrix(box_quat)
    corners_global = corners_local @ rot_global.T + np.array(box_center)
    corners_ego = transform_points(corners_global, ego_pose_inv[:3, :3], ego_pose_inv[:3, 3])
    corners_cam = transform_points(corners_ego, cam_extrin_inv[:3, :3], cam_extrin_inv[:3, 3])
    valid = corners_cam[:, 2] > 0
    if not np.any(valid):
        return None
    corners_cam = corners_cam[valid]
    xy = cam_intrin @ corners_cam.T
    xy = xy[:2, :] / xy[2, :]
    x_min, x_max = np.min(xy[0, :]), np.max(xy[0, :])
    y_min, y_max = np.min(xy[1, :]), np.max(xy[1, :])
    return (x_min, y_min, x_max, y_max)

# ========== 加载所有标定数据 ==========
print("加载标定数据...")
sensors = load_json('v1.0-mini/sensor.json')
sensor_token_to_channel = {s['token']: s['channel'] for s in sensors}

calib_sensors = load_json('v1.0-mini/calibrated_sensor.json')
calib_token_to_sensor = {}
calib_token_to_intrin = {}
for cs in calib_sensors:
    calib_token_to_sensor[cs['token']] = {
        'sensor_token': cs['sensor_token'],
        'rotation': cs['rotation'],
        'translation': cs['translation']
    }
    if 'camera_intrinsic' in cs:
        calib_token_to_intrin[cs['token']] = np.array(cs['camera_intrinsic'])

ego_poses = load_json('v1.0-mini/ego_pose.json')
ego_pose_map = {ep['token']: ep for ep in ego_poses}

sample_data_list = load_json('v1.0-mini/sample_data.json')
sample_data_by_token = {sd['token']: sd for sd in sample_data_list}

samples = load_json('v1.0-mini/sample.json')
sample_tokens = [s['token'] for s in samples]

# ========== 加载标注及类别映射 ==========
print("加载标注和类别...")
sample_anns = load_json('v1.0-mini/sample_annotation.json')
anns_by_sample = {}
for ann in sample_anns:
    token = ann['sample_token']
    anns_by_sample.setdefault(token, []).append(ann)

# 加载 instance.json 和 category.json 以获取类别名称
instances = load_json('v1.0-mini/instance.json')
instance_to_category = {inst['token']: inst['category_token'] for inst in instances}

categories = load_json('v1.0-mini/category.json')
cat_map = {c['token']: c['name'] for c in categories}

def get_category(ann):
    inst_token = ann.get('instance_token')
    if inst_token and inst_token in instance_to_category:
        cat_token = instance_to_category[inst_token]
        return cat_map.get(cat_token, 'unknown')
    return 'unknown'

# ========== 构建 sample -> 6 张图片路径 + 标定 ==========
print("构建样本映射...")
sample_to_data = {}
for token in sample_tokens:
    sample_to_data[token] = {'images': {}, 'calibs': {}}

for sd in sample_data_list:
    if not sd['is_key_frame']:
        continue
    calib_token = sd['calibrated_sensor_token']
    if calib_token not in calib_token_to_sensor:
        continue
    sensor_token = calib_token_to_sensor[calib_token]['sensor_token']
    channel = sensor_token_to_channel.get(sensor_token)
    if channel not in ORDER:
        continue
    sample_token = sd['sample_token']
    if sample_token not in sample_to_data:
        continue
    img_path = os.path.join(DATA_ROOT, sd['filename'])
    sample_to_data[sample_token]['images'][channel] = img_path

    # 保存标定
    ego = ego_pose_map[sd['ego_pose_token']]
    ego_mat = np.eye(4)
    ego_mat[:3, :3] = quaternion_to_rotation_matrix(ego['rotation'])
    ego_mat[:3, 3] = ego['translation']
    intrin = calib_token_to_intrin.get(calib_token)
    extrin = np.eye(4)
    extrin[:3, :3] = quaternion_to_rotation_matrix(calib_token_to_sensor[calib_token]['rotation'])
    extrin[:3, 3] = calib_token_to_sensor[calib_token]['translation']
    sample_to_data[sample_token]['calibs'][channel] = {
        'intrin': intrin,
        'extrin': extrin,
        'ego': ego_mat
    }

# 筛选完整样本
sample_list = []
for token, data in sample_to_data.items():
    if all(cam in data['images'] for cam in ORDER) and all(cam in data['calibs'] for cam in ORDER):
        paths = [data['images'][cam] for cam in ORDER]
        calibs = [data['calibs'][cam] for cam in ORDER]
        anns = anns_by_sample.get(token, [])
        sample_list.append((token, paths, calibs, anns))

print(f"✅ 找到 {len(sample_list)} 个有效样本（含全部6个相机和标定）")

# ========== 投影函数 ==========
def project_anns_to_image(anns, intrin, extrin, ego, img_shape=(900, 1600)):
    boxes_2d = []
    for ann in anns:
        center = ann['translation']
        size = ann['size']
        rot_quat = ann['rotation']
        cat = get_category(ann)   # 使用新函数
        box = project_3d_box_to_2d(center, size, rot_quat, intrin, extrin, ego)
        if box is not None:
            x1, y1, x2, y2 = box
            # 裁剪到图像范围
            x1 = max(0, min(x1, img_shape[1]-1))
            y1 = max(0, min(y1, img_shape[0]-1))
            x2 = max(0, min(x2, img_shape[1]-1))
            y2 = max(0, min(y2, img_shape[0]-1))
            if x2 > x1 and y2 > y1:
                boxes_2d.append((x1, y1, x2, y2, cat))
    return boxes_2d

# ========== 显示函数 ==========
def show_sample(paths, calibs, anns, sample_token, idx, total):
    orig_h, orig_w = 900, 1600  # nuScenes原始尺寸
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 2, figure=fig)
    titles = ['Front', 'Front Left', 'Front Right', 'Rear Left', 'Rear Right', 'Rear']
    # 创建输出目录
    output_dir = "samples_output_with_annotations"
    os.makedirs(output_dir, exist_ok=True) 
    for i, (path, calib) in enumerate(zip(paths, calibs)):
        if i == 0:
            ax = fig.add_subplot(gs[0, :])
        elif i == 1:
            ax = fig.add_subplot(gs[1, 0])
        elif i == 2:
            ax = fig.add_subplot(gs[1, 1])
        elif i == 3:
            ax = fig.add_subplot(gs[2, 0])
        elif i == 4:
            ax = fig.add_subplot(gs[2, 1])
        else:
            ax = fig.add_subplot(gs[3, :])
        
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(titles[i])
        ax.axis('off')
        
        intrin = calib['intrin']
        extrin = calib['extrin']
        ego = calib['ego']
        if intrin is not None:
            boxes = project_anns_to_image(anns, intrin, extrin, ego, img_shape=(orig_h, orig_w))
            for (x1, y1, x2, y2, cat) in boxes:
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, cat, color='red', fontsize=6,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    
    fig.suptitle(f"Sample {idx+1}/{total}  Token: {sample_token[:8]}...")
    fig.canvas.manager.set_window_title('nuScenes Viewer (Space: next, q: quit)')
    
    def on_key(event):
        if event.key == ' ':
            plt.close(fig)
        elif event.key == 'q':
            plt.close('all')
            exit(0)
    
#fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"samples_{idx+1:04d}_{sample_token[:8]}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

#plt.show()

# ========== 主循环 ==========
if __name__ == '__main__':
    total = len(sample_list)
    if total == 0:
        print("❌ 没有找到任何有效样本，请检查 DATA_ROOT 路径。")
        exit(0)
    idx = 0
    while True:
        token, paths, calibs, anns = sample_list[idx]
        show_sample(paths, calibs, anns, token, idx, total)
        idx = (idx + 1) % total
