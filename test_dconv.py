import convmodule
from funcs import conv, d_conv
import numpy as np

# 随机数据
N, C, H, W = 1, 1, 5, 5
C_out, Hk, Wk = 1, 3, 3
stride, padding = 1, 0

x = np.random.randn(N, C, H, W).astype(np.float32)
w = np.random.randn(C_out, C, Hk, Wk).astype(np.float32) * 0.1
b = np.random.randn(C_out).astype(np.float32)
result = conv(x, w, b, stride, padding)
if isinstance(result, tuple) and len(result) == 2:
    out, cache = result
else:
    out = result
    # 这里可以手动计算 x_pad，但为了简单，建议统一返回两个
# 前向（使用 NumPy 版本，获取缓存）
out,x_pad,(H_out,W_out) = conv(x, w, b, stride, padding)

# 随机上游梯度
dout = np.random.randn(N, C_out, H_out, W_out).astype(np.float32)

# C 扩展反向
#dx_c, dw_c, db_c = convmodule.d_conv_c(dout, x, w, stride, padding, x_pad, H_out, W_out)
dx_c, dw_c, db_c = convmodule.d_conv_c(dout, x, w, stride, padding, x_pad)

# NumPy 反向
dx_np, dw_np, db_np = d_conv(dout, x, w, stride, padding, x_pad)

# 对比
print("dx 差异:", np.abs(dx_c - dx_np).max())
print("dw 差异:", np.abs(dw_c - dw_np).max())
print("db 差异:", np.abs(db_c - db_np).max())
