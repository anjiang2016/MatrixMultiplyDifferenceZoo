import convmodule
import numpy as np
from funcs import conv as conv_numpy

# 随机输入
x = np.random.randn(1, 1, 5, 5).astype(np.float32)
w = np.random.randn(1, 1, 3, 3).astype(np.float32)

# C 扩展
out_c = convmodule.conv_c(x, w, 1, 0)

# NumPy 版本
out_np, _, _ = conv_numpy(x, w, stride=1, padding=0)

# 对比
diff = np.abs(out_c - out_np).max()
print(f"最大差异: {diff:.6f}")
print(f"C 扩展输出形状: {out_c.shape}")
print(f"NumPy 输出形状: {out_np.shape}")

