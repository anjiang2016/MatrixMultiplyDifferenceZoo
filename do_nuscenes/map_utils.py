import numpy as np
MAP_CONFIG = {
    # ===== 新加坡 =====
    'singapore-onenorth': {
        'image_path': 'maps/53992ee3023e5494b90c316c183be829.png',
        'utm_origin': [317215.0, 142698.0, 0.0],   # 地图左上角对应的 UTM 坐标
        'utm_offset': [317215.0, 142698.0, 0.0],   # translation 所需的偏移（与 utm_origin 相同）
        'img_h':20250.0,
        'utm_zone': 48,
        'resolution': 0.1,
        'description': 'Singapore One North'
    },
    'singapore-queenstown': {
        'image_path': 'maps/93406b464a165eaba6d9de76ca09f5da.png',
        'utm_origin': [317200.0, 143000.0, 0.0],
        'utm_offset': [317200.0, 143000.0, 0.0],
        'img_h':36871.0,
        'utm_zone': 48,
        'resolution': 0.1,
        'description': 'Singapore Holland Village'
    },
    'singapore-hollandvillage': {
        'image_path': 'maps/37819e65e09e5547b8a3ceaefba56bb2.png',
        'utm_origin': [317100.0, 141500.0, 0.0],
        'utm_offset': [317100.0, 141500.0, 0.0],
        'img_h':29229.0,
        'utm_zone': 48,
        'resolution': 0.1,
        'description': 'Singapore Queenstown'
    },
    # ===== 波士顿 =====
    'boston-seaport': {
        'image_path': 'maps/36092f0b03a857c6a3403e25b4b7aab3.png',
        'utm_origin': [322500.0, 4678500.0, 0.0],
        'utm_offset': [322500.0, 4678500.0, 0.0],
        'img_h':21181.0,
        'utm_zone': 19,
        'resolution': 0.1,
        'description': 'Boston Seaport'
    }
}
def utm_to_map_pixel(translation, city, map_config=MAP_CONFIG):
    """
    将 ego_pose.translation 转换为地图像素坐标
    
    Args:
        translation: [x, y, z] 来自 ego_pose.translation
        city: 城市名称
        map_config: 地图配置字典
    
    Returns:
        (col, row): 地图图像上的像素坐标
        utm_x, utm_y: 真实的 UTM 坐标（用于验证）
    """
    if city not in map_config:
        raise ValueError(f"未知城市: {city}")
    
    cfg = map_config[city]
    origin_x, origin_y, _ = cfg['utm_origin']   # 地图左上角的 UTM
    offset_x, offset_y, _ = cfg['utm_offset']   # 通常与 origin 相同
    res = cfg['resolution']
    
    # 真实 UTM 坐标 = translation + offset
    utm_x = translation[0] + offset_x
    utm_y = translation[1] + offset_y
    
    # 向量化计算（自动处理标量和数组）
    col = np.round((utm_x - origin_x) / res).astype(np.int64)
    row = np.round((utm_y - origin_y) / res).astype(np.int64)
    
    return col, row, utm_x, utm_y
def build_transform_matrix(city, offset_x=0.0, offset_y=0.0, map_config=MAP_CONFIG):
    """
    构建从全局坐标 (X_world, Y_world) 到显示图像素坐标的仿射变换矩阵
    
    Args:
        city: 城市名称
        offset_x, offset_y: 手动对齐偏移量（像素）
        map_config: 地图配置
    
    Returns:
        3x3 numpy 数组 (np.float64)
    """
    if city not in map_config:
        raise ValueError(f"未知城市: {city}")
    
    cfg = map_config[city]
    origin_x, origin_y, _ = cfg['utm_origin']
    res = cfg['resolution']
    offset_y = cfg['img_h']*res
    # 构建矩阵
    T = np.array([
        [1.0, 0.0,     offset_x ],
        [0.0,    -1.0, offset_y],
        [0.0,     0.0,     1.0]
    ], dtype=np.float64)
    return T
