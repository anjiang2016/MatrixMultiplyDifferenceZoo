import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

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

#R_cam_to_ego = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
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


# ========== 3. 预计算映射表 ==========
def build_lut(z_fixed=Z_FIXED):
    # ===== 1. 在地图坐标系下生成网格 =====
    # 地图坐标系：X 向右，Y 向下（车头朝上）
    x_map = np.arange(X_MIN, X_MAX, RES) 
    y_map = np.arange(X_MIN, X_MAX, RES)   
    xv, yv = np.meshgrid(x_map, y_map)

    X_map = xv.flatten()  # 地图 X
    Y_map = yv.flatten()  # 地图 Y
    # ===== 2. 定义地图坐标系 → 自车坐标系的变换矩阵 =====
    R_map_to_ego = np.array([
        [0, -1],
        [-1,  0]
    ])
    t_map_to_ego = np.array([X_MAX,Y_MAX])  # 原点重合，无平移
    # ===== 3. 用矩阵乘法将地图坐标映射到自车坐标 =====
    # [X_ego; Y_ego] = R_map_to_ego @ [X_map; Y_map] + t_map_to_ego
    map_coords = np.vstack([X_map, Y_map])  # (2, N)
    ego_coords = R_map_to_ego @ map_coords + t_map_to_ego.reshape(-1, 1)
    X_ego = ego_coords[0, :]
    Y_ego = ego_coords[1, :]

    Z = np.full_like(X_ego, z_fixed)
    ones = np.ones_like(X_ego)

    # ===== 3. 自车坐标 → 图像像素坐标 =====
    P_ego = np.vstack([X_ego, Y_ego, Z, ones])
    P_cam = Rt_full @ P_ego
    Zc = P_cam[2, :]

    pixel_raw = K @ P_cam
    u = pixel_raw[0, :] / Zc
    v = pixel_raw[1, :] / Zc

    # ===== 4. 返回地图坐标和像素坐标 =====
    grid_shape = (len(y_map), len(x_map))  # (行数, 列数)

    return u, v, grid_shape, X_map, Y_map, Z
def build_lut_ego(z_fixed=Z_FIXED):
    """生成查找表：BEV 每个网格点 (X, Y, z_fixed) 对应的 (u, v)"""
    
    x_coords = np.arange(X_MIN, X_MAX, RES)
    y_coords = np.arange(Y_MAX, Y_MIN, -RES)
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
            #row = int((Y[i] - Y_MIN) / RES)
            #col = int((X[i] - X_MIN) / RES)
            #车前朝右
#            bev_map[row, col] = color
            #车前朝上
            row = int(round((Y_MAX-Y[i]) / RES))
            col = int(round((X_MAX-X[i]) / RES))
            if 0<=row<map_h and 0<=col<map_w:
                bev_map[col,row] = color
    
    return bev_map
from scipy.ndimage import map_coordinates

def generate_bev_fast_(img, lut):
    import pdb;pdb.set_trace()
    u, v, grid_shape, X, Y, Z = lut
    bev_h, bev_w = grid_shape  # bev_h = 400 (Y方向), bev_w = 500 (X方向)

    u = u.astype(np.float64).flatten()
    v = v.astype(np.float64).flatten()
    coords = np.vstack([v,u])  # (2, N)

    # 对图像进行双线性插值采样（order=1），mode='nearest' 保证越界时取边缘值
    sampled = map_coordinates(img, coords, order=1, mode='nearest')  # (N, 3)

    # 重塑为你定义的形状 (bev_w, bev_h, 3) 并转为 uint8
    bev_map = sampled.reshape(bev_h, bev_w, 3).astype(np.uint8)

    return bev_map
from scipy.ndimage import map_coordinates

def generate_bev_fast(img, lut):
    u, v, grid_shape, X, Y, Z = lut
    bev_h, bev_w = grid_shape
    H, W, C = img.shape

    # ===== 关键修复：将图像展平为 2D =====
    # map_coordinates 对 3D (H,W,C) 图像要求坐标数组为 (3,N)
    # 我们将其展平为 (H, W*C)，变成 2D 图像，坐标数组只需 (2,N)
    img_flat = np.transpose(img,(0,2,1)).reshape(H, W * C)  # (900, 4800)
    # 1. 计算有效掩码
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    # ===== 裁剪坐标到有效范围 =====
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    # ===== 构造坐标数组 (2, N)，第一行是行(v)，第二行是列(u) =====
    #coords = np.vstack([v.astype(np.float64), u.astype(np.float64)])
    # 2. 构造坐标数组 (3, N)：每一列为 [v, u, channel]
    N = len(u)
    coords = np.zeros((2, N), dtype=np.float64)
    coords[0] = v  # 行坐标
    coords[1] = u  # 列坐标
    # 第三维留空，需要扩展为 3N
    coords = np.repeat(coords, C, axis=1)  # (3, N*C)
    offsets = np.tile(np.arange(C)*W,N)
    coords[1] += offsets
    # ===== 双线性插值 =====
    #sampled_flat = map_coordinates(img_flat, coords, order=1, mode='nearest')  # (N,)
    sampled_flat = map_coordinates(img_flat, coords, order=1, mode='constant',cval=0)  # (N,)
    # ===== 恢复为 3 通道 =====
    sampled = sampled_flat.reshape(-1, C)  # (N, 3)
    # ===== 重塑为 BEV 网格 (bev_h, bev_w, 3) =====
    # 这与你的原始 for 循环完全一致：行对应 Y，列对应 X，车头向右
    bev_map = sampled.reshape(bev_h, bev_w, C).astype(np.uint8)
    bev_map = bev_map* mask.reshape(bev_h, bev_w, 1).astype(np.uint8)
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
    bev = generate_bev_fast(img_np, lut_fixed)
    print("BEV 生成完成")
    # 将 BEV 图缩放到与原图高度一致（保持宽高比）
    h_orig = img_np.shape[0]
    w_bev = int(bev.shape[1] * h_orig / bev.shape[0])
    bev_resized = np.array(Image.fromarray(bev).resize((w_bev, h_orig)))
        
    # 左右拼接
    combined = np.hstack([img_np, bev_resized])
    # 保存 BEV 图（PIL 保存）
    bev_pil = Image.fromarray(combined)
    bev_pil.save("bev_output.jpg")
    print("BEV 已保存为 bev_output.jpg")

    # 显示 BEV 图（调用系统默认图片查看器）
    bev_pil.show()
