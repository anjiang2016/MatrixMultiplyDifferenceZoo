# view_nuscenes_samples.py
import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
# ========== 配置 ==========
DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"   # 改成你的实际路径
ORDER = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
         'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
# ========== 加载 JSON ==========
def load_json(rel_path):
    with open(os.path.join(DATA_ROOT, rel_path), 'r') as f:
        return json.load(f)

# 1. sensor.json：sensor_token -> channel
sensors = load_json('v1.0-mini/sensor.json')
sensor_token_to_channel = {s['token']: s['channel'] for s in sensors}

# 2. calibrated_sensor.json：calibrated_sensor_token -> sensor_token
calib_sensors = load_json('v1.0-mini/calibrated_sensor.json')
calib_token_to_sensor_token = {cs['token']: cs['sensor_token'] for cs in calib_sensors}

# 3. sample_data.json：提取关键帧图像
sample_data_list = load_json('v1.0-mini/sample_data.json')

# 4. sample.json：获取所有 sample token
samples = load_json('v1.0-mini/sample.json')
sample_tokens = [s['token'] for s in samples]

# ========== 构建 sample_token -> 6张图片路径 ==========
sample_to_images = {token: {} for token in sample_tokens}

for sd in sample_data_list:
    if not sd['is_key_frame']:
        continue
    calib_token = sd['calibrated_sensor_token']
    sensor_token = calib_token_to_sensor_token.get(calib_token)
    if sensor_token is None:
        continue
    channel = sensor_token_to_channel.get(sensor_token)
    if channel in ORDER:
        sample_token = sd['sample_token']
        if sample_token in sample_to_images:
            img_path = os.path.join(DATA_ROOT, sd['filename'])
            sample_to_images[sample_token][channel] = img_path

# 整理为列表（只保留6个相机都存在的样本）
sample_list = []
for token, cam_dict in sample_to_images.items():
    if all(cam in cam_dict for cam in ORDER):
        paths = [cam_dict[cam] for cam in ORDER]
        sample_list.append((token, paths))

print(f"✅ 找到 {len(sample_list)} 个有效样本（含全部6个相机）")

# ========== 显示函数 ==========
def show_sample(paths, sample_token, idx, total):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 2, figure=fig)   # 4行2列

    # 布局：前摄像头跨两列（占第一行）
    ax1 = fig.add_subplot(gs[0, :])
    img = mpimg.imread(paths[0])
    ax1.imshow(img)
    ax1.set_title('Front')
    ax1.axis('off')

    # 第二行：前左、前右
    ax2 = fig.add_subplot(gs[1, 0])
    img = mpimg.imread(paths[1])
    ax2.imshow(img)
    ax2.set_title('Front Left')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 1])
    img = mpimg.imread(paths[2])
    ax3.imshow(img)
    ax3.set_title('Front Right')
    ax3.axis('off')

    # 第三行：后左、后右
    ax4 = fig.add_subplot(gs[2, 0])
    img = mpimg.imread(paths[3])
    ax4.imshow(img)
    ax4.set_title('Rear Left')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[2, 1])
    img = mpimg.imread(paths[4])
    ax5.imshow(img)
    ax5.set_title('Rear Right')
    ax5.axis('off')

    # 第四行：后摄像头跨两列
    ax6 = fig.add_subplot(gs[3, :])
    img = mpimg.imread(paths[5])
    ax6.imshow(img)
    ax6.set_title('Rear')
    ax6.axis('off')

    fig.suptitle(f"Sample {idx+1}/{total}  Token: {sample_token[:8]}...")
    fig.canvas.manager.set_window_title('nuScenes Viewer (Space: next, q: quit)')

    def on_key(event):
        if event.key == ' ':
            plt.close(fig)
        elif event.key == 'q':
            plt.close('all')
            exit(0)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()
# ========== 主循环 ==========
if __name__ == '__main__':
    total = len(sample_list)
    if total == 0:
        print("❌ 没有找到任何有效样本，请检查 DATA_ROOT 路径是否正确。")
        exit(0)
    idx = 0
    while True:
        token, paths = sample_list[idx]
        show_sample(paths, token, idx, total)
        idx = (idx + 1) % total
