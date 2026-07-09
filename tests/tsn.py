import numpy as np                                                                                                                                   
import pickle
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

# 导入所有层函数
from funcs import (
    conv, d_conv,
    sigmoid, d_sigmoid,
    relu,d_relu,
    avgpool, d_avgpool,
    flatten, d_flatten,
    linear, d_linear,
    softmax, d_softmax,
    softmax_cross_entropy, d_softmax_cross_entropy
)

def test_simple_network():
    """测试最简单的单层网络"""
    print("="*60)
    print("测试单层 Linear 网络")
    print("="*60)
    
    N, D_in, D_out = 10, 128, 10
    
    # 随机数据
    x = np.random.randn(N, D_in).astype(np.float32)
    y = np.random.randint(0, D_out, size=N)
    w = np.random.randn(D_in, D_out).astype(np.float32) * 0.01
    b = np.zeros(D_out, dtype=np.float32)
    
    # 前向
    logits, cache = linear(x, w, b)
    loss, ce_cache = softmax_cross_entropy(logits, y)
    print(f"初始 Loss: {loss:.6f}")
    
    # 梯度检查
    dlogits = d_softmax_cross_entropy(ce_cache)
    dx, dw, db = d_linear(dlogits, cache)
    
    # 数值梯度
    eps = 1e-5
    dw_num = np.zeros_like(w)
    for i in range(min(5, D_in)):
        for j in range(min(5, D_out)):
            w_plus = w.copy()
            w_minus = w.copy()
            w_plus[i, j] += eps
            w_minus[i, j] -= eps
            
            logits_plus, _ = linear(x, w_plus, b)
            logits_minus, _ = linear(x, w_minus, b)
            
            loss_plus, _ = softmax_cross_entropy(logits_plus, y)
            loss_minus, _ = softmax_cross_entropy(logits_minus, y)
            
            dw_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    print(f"解析梯度 dw[0,0]: {dw[0,0]:.8f}")
    print(f"数值梯度 dw[0,0]: {dw_num[0,0]:.8f}")
    print(f"差异: {abs(dw[0,0] - dw_num[0,0]):.8f}")
    
    # 训练几步
    print("\n训练 100 步...")
    for step in range(100):
        logits, cache = linear(x, w, b)
        loss, ce_cache = softmax_cross_entropy(logits, y)
        dlogits = d_softmax_cross_entropy(ce_cache)
        dx, dw, db = d_linear(dlogits, cache)
        
        w -= 0.1 * dw
        b -= 0.1 * db
        
        if (step + 1) % 20 == 0:
            acc = np.mean(np.argmax(logits, axis=1) == y)
            print(f"  Step {step+1}: Loss={loss:.6f}, Acc={acc:.4f}")
    
    print("\n✅ 如果 Loss 下降，说明基础层是正确的")
    print("   问题可能在 Conv 或 Pool 层")
def test_conv_network():
    """测试单层 Conv 网络"""
    print("\n" + "="*60)
    print("测试单层 Conv 网络")
    print("="*60)
    
    N, C, H, W = 10, 1, 32, 32
    C_out = 6
    Hk, Wk = 5, 5
    
    x = np.random.randn(N, C, H, W).astype(np.float32)
    y = np.random.randint(0, 10, size=N)
    w = np.random.randn(C_out, C, Hk, Wk).astype(np.float32) * 0.01
    b = np.zeros(C_out, dtype=np.float32)
    
    # 前向: Conv -> Flatten -> Linear -> Softmax
    out, x_pad, (H_out, W_out) = conv(x, w, b, stride=1, padding=0)
    flat = out.reshape(N, -1)
    D_flat = flat.shape[1]
    w2 = np.random.randn(D_flat, 10).astype(np.float32) * 0.01
    b2 = np.zeros(10, dtype=np.float32)
    
    logits, cache = linear(flat, w2, b2)
    loss, ce_cache = softmax_cross_entropy(logits, y)
    print(f"初始 Loss: {loss:.6f}")
    
    # 训练
    print("\n训练 100 步...")
    for step in range(100):
        out, x_pad, (H_out, W_out) = conv(x, w, b, stride=1, padding=0)
        flat = out.reshape(N, -1)
        logits, cache = linear(flat, w2, b2)
        loss, ce_cache = softmax_cross_entropy(logits, y)
        
        dlogits = d_softmax_cross_entropy(ce_cache)
        dx, dw2, db2 = d_linear(dlogits, cache)
        w2 -= 0.1 * dw2
        b2 -= 0.1 * db2
        
        # 需要实现 Conv 的反向传播
        # ... 
        
        if (step + 1) % 20 == 0:
            acc = np.mean(np.argmax(logits, axis=1) == y)
            print(f"  Step {step+1}: Loss={loss:.6f}, Acc={acc:.4f}")
if __name__ == "__main__":
    test_simple_network()
    test_conv_network()
