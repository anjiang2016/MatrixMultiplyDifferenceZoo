import numpy as np
import time
import sys

# 确保当前目录在 sys.path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import convmodule

print("开始测试...")
M = 4096 
N = 4096
K = 4096

A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

# CPU 参考
start = time.time()
C_cpu = A @ B
cpu_time = (time.time() - start) * 1000
print(f"CPU 时间: {cpu_time:.2f} ms")

# Metal 测试
print("调用 convmodule.test_matmul ...")
C_metal, metal_time = convmodule.test_matmul(A, B)
print(f"Metal 时间: {metal_time} ms")

# 比较
diff = np.abs(C_cpu - C_metal).max()
print(f"最大差异: {diff:.6f}")

if diff < 1e-4:
    print("✅ Metal 结果正确！")
    if metal_time > 0 and cpu_time > 0:
        speedup = cpu_time / metal_time
        print(f"加速比: {speedup:.2f}x")
else:
    print("❌ Metal 结果错误！")
