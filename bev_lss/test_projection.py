# test_projection.py
import json
import numpy as np
import os

DATA_ROOT = "/Users/zhaomingming/data_sets/v1.0-mini"
CAM = 'CAM_FRONT'

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

sensors = load_json('v1.0-mini/sensor.json')
sensor_token_to_channel = {s['token']: s['channel'] for s in sensors}

calib_sensors = load_json('v1.0-mini/calibrated_sensor.json')
calib_map = {}
for cs in calib_sensors:
    sensor_token = cs['sensor_token']
    if sensor_token not in sensor_token_to_channel:
        continue
    channel = sensor_token_to_channel[sensor_token]
    if channel != CAM:
        continue
    intrin = np.array(cs.get('camera_intrinsic', np.eye(3)))
    R = quaternion_to_rotation_matrix(cs['rotation'])
    t = np.array(cs['translation'])
    calib_map[cs['token']] = {
        'channel': channel,
        'intrin': intrin,
        'R_sensor2ego': R,      # 直接使用，不取逆
        't_sensor2ego': t
    }

samples = load_json('v1.0-mini/sample.json')
sample_token = samples[0]['token']
sample_data_list = load_json('v1.0-mini/sample_data.json')
target_calib_token = None
for sd in sample_data_list:
    if sd['sample_token'] == sample_token and sd['is_key_frame']:
        calib_token = sd['calibrated_sensor_token']
        if calib_token in calib_map:
            target_calib_token = calib_token
            break

cam_info = calib_map[target_calib_token]
intrin = cam_info['intrin']
R_sensor2ego = cam_info['R_sensor2ego']
t_sensor2ego = cam_info['t_sensor2ego']

print("Intrinsic:\n", intrin)
print("R_sensor2ego:\n", R_sensor2ego)
print("t_sensor2ego:", t_sensor2ego)

# 图像中心点
u0, v0 = 800, 450
fx, fy = intrin[0,0], intrin[1,1]
cx, cy = intrin[0,2], intrin[1,2]
x_c = (u0 - cx) / fx
y_c = (v0 - cy) / fy
ray = np.array([x_c, y_c, 1.0])

# 求地面交点：P_ego = R_sensor2ego @ (lambda * ray) + t_sensor2ego, 令 z=0
denom = R_sensor2ego[2,0]*ray[0] + R_sensor2ego[2,1]*ray[1] + R_sensor2ego[2,2]*ray[2]
if abs(denom) < 1e-8:
    print("无交点")
else:
    lam = -t_sensor2ego[2] / denom
    P_ego = lam * (R_sensor2ego @ ray) + t_sensor2ego
    print("地面交点 (x, y):", P_ego[0], P_ego[1])
