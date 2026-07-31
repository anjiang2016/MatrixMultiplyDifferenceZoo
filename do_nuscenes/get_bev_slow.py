import numpy as np
from PIL import Image

# ========== 1. nuScenes 真实标定参数 ==========

K = np.array([
    [1266.417203046554, 0.0,                816.2670197447984],
    [0.0,                1266.417203046554,  491.50706579294757],
    [0.0,                0.0,                1.0]
])

t = np.array([[1.70079119],[0.01594563],[1.51095764]])

quat = [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755]
# ========== 8. 四元数 → 旋转矩阵 ==========
def quat_to_rot(w, x, y, z):
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

R_cam_to_ego = quat_to_rot(*quat)
R_ego_to_cam = np.linalg.inv(R_cam_to_ego)
# 完整外参矩阵 [R, t] (3×4)
Rt_full = np.hstack([R_ego_to_cam,-R_ego_to_cam @ t])

IMG_H, IMG_W = 900, 1600


# ========== 2. BEV 网格参数 ==========

X_MIN, X_MAX = 0.0, 40.0
Y_MIN, Y_MAX = -20.0, 20.0
Z_FIXED = 0.0               # 默认地面高度，可改成任意值
RES = 0.1

def build_lut(z_fixed=Z_FIXED):
    """生成查找表：BEV 每个网格点 (X, Y, z_fixed) 对应的 (u, v)"""
    
    x_coords = np.arange(X_MIN, X_MAX, RES)
    y_coords = np.arange(Y_MIN, Y_MAX, RES)
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    X = xv.flatten()
    Y = yv.flatten()
    Z = np.full_like(X, z_fixed)
    ones = np.ones_like(X)
    
    P_ego = np.vstack([X, Y, Z, ones])
    P_cam = Rt_full @ P_ego
    Zc = P_cam[2, :]
    
    pixel_raw = K @ P_cam
	
    u_raw = pixel_raw[0, :]
    v_raw = pixel_raw[1, :]

    #u_raw = pixel_raw[1, :]  # 原来存的是行坐标（v），现在当成列坐标（u）
    #v_raw = pixel_raw[0, :]  # 原来存的是列坐标（u），现在当成行坐标（v） 

    u = u_raw / Zc
    v = v_raw / Zc
    
    grid_shape = (len(y_coords), len(x_coords))
    return u, v, grid_shape, X, Y, Z


# ========== 4. 双线性插值（与之前相同） ==========

def bilinear_sample(img, u, v):
    h, w = img.shape[:2]
    
    if u < 0 or u >= w or v < 0 or v >= h:
        return np.array([0, 0, 0], dtype=np.uint8)
    
    u0, v0 = int(np.floor(u)), int(np.floor(v))
    u1, v1 = min(u0 + 1, w - 1), min(v0 + 1, h - 1)
    
    f00 = img[v0, u0].astype(np.float32)
    f10 = img[v0, u1].astype(np.float32)
    f01 = img[v1, u0].astype(np.float32)
    f11 = img[v1, u1].astype(np.float32)
    
    du, dv = u - u0, v - v0
    top = (1 - du) * f00 + du * f10
    bottom = (1 - du) * f01 + du * f11
    color = (1 - dv) * top + dv * bottom
    
    return color.astype(np.uint8)


# ========== 5. 查表生成 BEV ==========

def generate_bev(img, lut):
    u, v, grid_shape, X, Y, Z = lut
    bev_h, bev_w = grid_shape
#    //车前朝右
#    bev_map = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    #车前朝上
    bev_map = np.zeros((bev_w, bev_h, 3), dtype=np.uint8)
    map_h,map_w = int((Y_MAX-Y_MIN)/RES),int((X_MAX-X_MIN)/RES)
    for i in range(len(u)):
        ui, vi = u[i], v[i]
        if 0 <= ui < IMG_W and 0 <= vi < IMG_H:
            color = bilinear_sample(img, ui, vi)
            row = int(round((Y[i] - Y_MIN) / RES))
            col = int(round((X[i] - X_MIN) / RES))
            #车前朝右
            bev_map[row, col] = color
            #if 0<=row<map_h and 0<=col<map_w:
            #    bev_map[col,row] = color
    
    return bev_map


# ========== 6. 支持传入自定义高度 ==========

def build_lut_with_height(X, Y, Z):
    """直接传入任意点集 (X, Y, Z)，生成对应的查找表"""
    ones = np.ones_like(X)
    P_ego = np.vstack([X, Y, Z, ones])
    P_cam = Rt_full @ P_ego
    Zc = P_cam[2, :]
    pixel_raw = K @ P_cam
    u = pixel_raw[0, :] / Zc
    v = pixel_raw[1, :] / Zc
    return u, v


# ========== 7. 使用示例（PIL 版本） ==========

if __name__ == "__main__":
    # 预计算查找表
    lut_fixed = build_lut(z_fixed=0.0)
    print("查找表构建完成（Z=0）")

    # 使用 PIL 读取图片（自动转 RGB，无需 BGR 转换）
    try:
        img_pil = Image.open("/Users/zhaomingming/data_sets/v1.0-mini/sweeps/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927762460.jpg")
        img_np = np.array(img_pil)  # 转换为 NumPy 数组 (H, W, 3) RGB
    except FileNotFoundError:
        print("未找到图片，跳过生成")
        exit()

    # 生成 BEV
    bev = generate_bev(img_np, lut_fixed)
    print("BEV 生成完成")
    # 将 BEV 图缩放到与原图高度一致（保持宽高比）
    w_orig = img_np.shape[1]
    h_bev = int(bev.shape[0] * w_orig / bev.shape[1])
    bev_resized = np.array(Image.fromarray(bev).resize((w_orig, h_bev)))
        
    # 左右拼接
    combined = np.vstack([img_np, bev_resized])
    # 保存 BEV 图（PIL 保存）
    bev_pil = Image.fromarray(combined)
    bev_pil.save("bev_output.jpg")
    print("BEV 已保存为 bev_output.jpg")

    # 显示 BEV 图（调用系统默认图片查看器）
    bev_pil.show()
