import pyproj

# 真实经纬度（来自苹果地图）
lon, lat = 103.789934, 1.305407

# 新加坡 UTM zone 48N
utm_proj = pyproj.Proj(proj='utm', zone=48, ellps='WGS84', south=False)
utm_x, utm_y = utm_proj(lon, lat)

# 你之前获取的 translation
translation = [411.4199861830012, 1181.197175631848]

# 计算偏移量
offset_x = utm_x - translation[0]
offset_y = utm_y - translation[1]

print(f"✅ 正确的 UTM 偏移量 (可硬编码到脚本中):")
print(f"UTM_OFFSET = [{offset_x:.3f}, {offset_y:.3f}, 0.0]")
