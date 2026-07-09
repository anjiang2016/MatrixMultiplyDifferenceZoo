import convmodule
import numpy as np

x = np.random.randn(1, 1, 5, 5).astype(np.float32)
w = np.random.randn(1, 1, 3, 3).astype(np.float32)

# 直接调用 C 扩展（无偏置）
out = convmodule.conv_c(x, w, 1, 0)
print(out.shape)  # 应该是 (1, 1, 3, 3)
