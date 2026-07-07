import numpy as np
from funcs import linear, d_linear

# 创建数据
N, D_in, D_out = 4, 10, 5
x = np.random.randn(N, D_in).astype(np.float32)
w = np.random.randn(D_in, D_out).astype(np.float32) * 0.01
b = np.random.randn(D_out).astype(np.float32)

# 前向
out, cache = linear(x, w, b)
print(f"输出: {out.shape}")  # (4, 5)

# 反向
dout = np.random.randn(*out.shape).astype(np.float32)
dx, dw, db = d_linear(dout, cache)

print(f"dx: {dx.shape}")  # (4, 10)
print(f"dw: {dw.shape}")  # (10, 5)
print(f"db: {db.shape}")  # (5,)
