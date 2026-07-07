import numpy as np
import time

# 导入 flatten 函数
from flatten import flatten, d_flatten


def test_basic():
    """基础功能测试"""
    print("=" * 60)
    print("测试1: 基础功能测试")
    print("=" * 60)
    
    # 测试1: 4D 输入 (N, C, H, W)
    print("\n[1] 4D 输入 (N, C, H, W)")
    x1 = np.random.randn(2, 3, 28, 28).astype(np.float32)
    out1, cache1 = flatten(x1)
    print(f"  输入形状: {x1.shape}")
    print(f"  输出形状: {out1.shape}")
    print(f"  期望: (2, 2352)")  # 3*28*28 = 2352
    
    dout1 = np.random.randn(*out1.shape).astype(np.float32)
    dx1 = d_flatten(dout1, cache1)
    print(f"  梯度形状: {dx1.shape}")
    print(f"  是否正确还原: {x1.shape == dx1.shape}")
    
    # 测试2: 2D 输入 (N, D)
    print("\n[2] 2D 输入 (N, D)")
    x2 = np.random.randn(2, 128).astype(np.float32)
    out2, cache2 = flatten(x2)
    print(f"  输入形状: {x2.shape}")
    print(f"  输出形状: {out2.shape}")
    
    dout2 = np.random.randn(*out2.shape).astype(np.float32)
    dx2 = d_flatten(dout2, cache2)
    print(f"  梯度形状: {dx2.shape}")
    print(f"  是否正确还原: {x2.shape == dx2.shape}")
    
    # 测试3: 1D 输入 (无 batch 维度)
    print("\n[3] 1D 输入 (无 batch 维度)")
    x3 = np.random.randn(128).astype(np.float32)
    out3, cache3 = flatten(x3)
    print(f"  输入形状: {x3.shape}")
    print(f"  输出形状: {out3.shape}")
    print(f"  期望: (1, 128)")
    
    dout3 = np.random.randn(*out3.shape).astype(np.float32)
    dx3 = d_flatten(dout3, cache3)
    print(f"  梯度形状: {dx3.shape}")
    print(f"  是否正确还原: {x3.shape == dx3.shape}")
    
    # 测试4: 5D 输入
    print("\n[4] 5D 输入 (N, C, D, H, W)")
    x4 = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    out4, cache4 = flatten(x4)
    print(f"  输入形状: {x4.shape}")
    print(f"  输出形状: {out4.shape}")
    print(f"  期望: (2, 360)")  # 3*4*5*6 = 360
    
    dout4 = np.random.randn(*out4.shape).astype(np.float32)
    dx4 = d_flatten(dout4, cache4)
    print(f"  梯度形状: {dx4.shape}")
    print(f"  是否正确还原: {x4.shape == dx4.shape}")
    
    # 测试5: 空 batch 维度 (0)
    print("\n[5] 空 batch 维度 (0)")
    x5 = np.random.randn(0, 3, 28, 28).astype(np.float32)
    out5, cache5 = flatten(x5)
    print(f"  输入形状: {x5.shape}")
    print(f"  输出形状: {out5.shape}")
    print(f"  期望: (0, 2352)")
    
    dout5 = np.random.randn(*out5.shape).astype(np.float32)
    dx5 = d_flatten(dout5, cache5)
    print(f"  梯度形状: {dx5.shape}")
    print(f"  是否正确还原: {x5.shape == dx5.shape}")


def test_gradient():
    """梯度正确性验证（数值梯度检查）"""
    print("\n" + "=" * 60)
    print("测试2: 梯度正确性验证")
    print("=" * 60)
    
    # 创建一个包含 Flatten 的简单网络
    # 输入 -> Flatten -> Linear -> Loss
    N, C, H, W = 2, 3, 4, 4
    D = C * H * W
    
    x = np.random.randn(N, C, H, W).astype(np.float32)
    W_linear = np.random.randn(D, 10).astype(np.float32)
    
    print(f"\n网络结构: 输入 {x.shape} -> Flatten -> Linear(10) -> Loss")
    
    # 前向
    flat, cache = flatten(x)
    out = flat @ W_linear
    loss = (out ** 2).sum()
    
    print(f"  展平后: {flat.shape}")
    print(f"  输出: {out.shape}")
    print(f"  Loss: {loss:.6f}")
    
    # 反向（手动计算）
    dloss = 2 * out  # loss 对输出的梯度
    dflat = dloss @ W_linear.T  # 通过 Linear 反向
    dx = d_flatten(dflat, cache)  # 通过 Flatten 反向
    
    print(f"\n  梯度形状: {dx.shape}")
    
    # 数值梯度检查
    print("\n数值梯度检查 (对输入 x 的梯度):")
    eps = 1e-6
    
    # 检查所有元素（小数据量）
    dx_num = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[n, c, h, w] += eps
                    x_minus[n, c, h, w] -= eps
                    
                    flat_plus, _ = flatten(x_plus)
                    flat_minus, _ = flatten(x_minus)
                    
                    out_plus = flat_plus @ W_linear
                    out_minus = flat_minus @ W_linear
                    
                    loss_plus = (out_plus ** 2).sum()
                    loss_minus = (out_minus ** 2).sum()
                    
                    dx_num[n, c, h, w] = (loss_plus - loss_minus) / (2 * eps)
    
    max_diff = np.abs(dx - dx_num).max()
    print(f"  最大差异: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("  ✅ 梯度检查通过!")
    else:
        print("  ❌ 梯度检查失败!")


def test_performance():
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试3: 性能测试")
    print("=" * 60)
    
    # 不同尺寸的测试
    test_cases = [
        (1, 1, 28, 28, "1x1x28x28"),
        (64, 3, 32, 32, "64x3x32x32"),
        (128, 64, 16, 16, "128x64x16x16"),
        (256, 128, 8, 8, "256x128x8x8"),
    ]
    
    print("\n前向传播性能 (10000次):")
    print("-" * 60)
    print(f"{'输入形状':<25} {'时间':<12} {'输出维度':<15}")
    print("-" * 60)
    
    for N, C, H, W, name in test_cases:
        x = np.random.randn(N, C, H, W).astype(np.float32)
        
        start = time.time()
        for _ in range(10000):
            out, cache = flatten(x)
        elapsed = time.time() - start
        
        print(f"{name:<25} {elapsed:.4f}s    {N}x{C*H*W:<8}")
    
    print("\n反向传播性能 (10000次):")
    print("-" * 60)
    print(f"{'输入形状':<25} {'时间':<12}")
    print("-" * 60)
    
    for N, C, H, W, name in test_cases:
        x = np.random.randn(N, C, H, W).astype(np.float32)
        out, cache = flatten(x)
        dout = np.random.randn(*out.shape).astype(np.float32)
        
        start = time.time()
        for _ in range(10000):
            dx = d_flatten(dout, cache)
        elapsed = time.time() - start
        
        print(f"{name:<25} {elapsed:.4f}s")


def test_consistency():
    """一致性测试：确保 flatten 和 d_flatten 互为逆操作"""
    print("\n" + "=" * 60)
    print("测试4: 一致性测试")
    print("=" * 60)
    
    # 测试不同形状
    shapes = [
        (2, 3, 28, 28),
        (2, 128),
        (128,),
        (2, 3, 4, 5, 6),
        (0, 3, 28, 28),
    ]
    
    print("\n测试 flatten 和 d_flatten 是否互为逆操作:")
    print("-" * 60)
    
    for shape in shapes:
        x = np.random.randn(*shape).astype(np.float32)
        
        # 前向
        out, cache = flatten(x)
        
        # 反向（用 out 本身作为梯度）
        dx = d_flatten(out, cache)
        
        # 检查是否还原
        is_same = np.allclose(x, dx)
        
        print(f"  输入形状: {str(shape):<20} 还原: {'✅ 成功' if is_same else '❌ 失败'}")


def test_edge_cases():
    """边缘情况测试"""
    print("\n" + "=" * 60)
    print("测试5: 边缘情况测试")
    print("=" * 60)
    
    # 测试1: 非常大的维度
    print("\n[1] 大维度测试")
    x1 = np.random.randn(2, 1024, 1024).astype(np.float32)
    out1, cache1 = flatten(x1)
    print(f"  输入: {x1.shape} -> 输出: {out1.shape}")
    print(f"  期望: (2, 1048576)")
    
    # 测试2: 极小维度
    print("\n[2] 极小维度测试")
    x2 = np.random.randn(1, 1, 1, 1).astype(np.float32)
    out2, cache2 = flatten(x2)
    print(f"  输入: {x2.shape} -> 输出: {out2.shape}")
    print(f"  期望: (1, 1)")
    
    # 测试3: 全是 0
    print("\n[3] 全零输入测试")
    x3 = np.zeros((2, 3, 28, 28), dtype=np.float32)
    out3, cache3 = flatten(x3)
    print(f"  输入: {x3.shape}, 输出: {out3.shape}")
    print(f"  是否全零: {np.all(out3 == 0)}")
    
    dout3 = np.ones_like(out3)
    dx3 = d_flatten(dout3, cache3)
    print(f"  梯度是否全1: {np.all(dx3 == 1)}")
    print(f"  梯度形状: {dx3.shape}")
    
    # 测试4: 负数
    print("\n[4] 负数输入测试")
    x4 = np.random.randn(2, 3, 28, 28).astype(np.float32) * -1
    out4, cache4 = flatten(x4)
    print(f"  输入: {x4.shape}, 输出: {out4.shape}")
    print(f"  是否完全一致: {np.allclose(out4, x4.reshape(2, -1))}")


def test_compare_with_numpy():
    """与 NumPy 原生 reshape 对比"""
    print("\n" + "=" * 60)
    print("测试6: 与 NumPy reshape 对比")
    print("=" * 60)
    
    shapes = [
        (2, 3, 28, 28),
        (2, 128),
        (128,),
        (2, 3, 4, 5, 6),
    ]
    
    print("\n对比 flatten 和 numpy.reshape:")
    print("-" * 60)
    
    for shape in shapes:
        x = np.random.randn(*shape).astype(np.float32)
        
        # 我们的实现
        out_our, cache = flatten(x)
        
        # NumPy 原生
        if x.ndim == 1:
            out_np = x.reshape(1, -1)
        else:
            out_np = x.reshape(x.shape[0], -1)
        
        is_same = np.allclose(out_our, out_np)
        
        print(f"  输入: {str(shape):<20} 输出: {str(out_our.shape):<15} 匹配: {'✅' if is_same else '❌'}")


if __name__ == "__main__":
    # 运行所有测试
    test_basic()
    test_gradient()
    test_performance()
    test_consistency()
    test_edge_cases()
    test_compare_with_numpy()
    
    print("\n" + "=" * 60)
    print("所有测试完成! ✅")
    print("=" * 60)
