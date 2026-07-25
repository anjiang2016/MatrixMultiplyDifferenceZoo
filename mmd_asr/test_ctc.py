import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from funcs import ctc_loss, d_ctc_loss, softmax
import numpy as np
# 假设你有 logits (T=20, N=2, C=10), targets (2, 5)
T, N, C = 20, 2, 10
logits = np.random.randn(T, N, C).astype(np.float32)
targets = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
input_lengths = np.array([20, 18])
target_lengths = np.array([5, 4])

loss, cache = ctc_loss(logits, targets, input_lengths, target_lengths, blank=0)
print(f"CTC Loss: {loss:.6f}")

# 反向计算梯度
dlogits = d_ctc_loss(1.0, cache, logits)
print(f"Gradient shape: {dlogits.shape}")
