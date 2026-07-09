import numpy as np
import time

from funcs import softmax, d_softmax, softmax_cross_entropy, d_softmax_cross_entropy


def test_basic():
    """基础功能测试"""
    print("=" * 60)
    print("测试1: 基础功能测试")
    print("=" * 60)
    
    # 测试1: 标准 2D 输入 (N, D)
    print("\n[1] 标准 2D 输入 (N, D)")
    N, D = 3, 5
    x = np.random.randn(N, D).astype(np.float32)
    print(f"  输入: {x.shape}")
    print(f"  输入范围: [{x.min():.4f}, {x.max():.4f}]")
    
    out, cache = softmax(x)
    print(f"  输出: {out.shape}")
    print(f"  输出范围: [{out.min():.6f}, {out.max():.6f}]")
    print(f"  每行和: {out.sum(axis=1)}")
    print(f"  是否和为1: {np.allclose(out.sum(axis=1), 1.0)}")
    
    # 反向
    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = d_softmax(dout, cache)
    print(f"  梯度: {dx.shape}")
    print(f"  梯度范围: [{dx.min():.4f}, {dx.max():.4f}]")
    
    # 测试2: 3D 输入 (N, C, H)
    print("\n[2] 3D 输入 (N, C, H)")
    x3d = np.random.randn(2, 3, 4).astype(np.float32)
    out3d, cache3d = softmax(x3d, axis=1)  # 在 channel 维度做 softmax
    print(f"  输入: {x3d.shape}")
    print(f"  输出: {out3d.shape}")
    print(f"  每层和 (axis=1): {out3d.sum(axis=1)}")
    print(f"  是否和为1: {np.allclose(out3d.sum(axis=1), 1.0)}")
    
    # 测试3: 数值稳定性（大数值）
    print("\n[3] 数值稳定性测试（大数值）")
    x_large = np.array([[1000, 1000, 1000], 
                        [1000, 1001, 1000]], dtype=np.float32)
    out_large, _ = softmax(x_large)
    print(f"  输入: \n{x_large}")
    print(f"  输出: \n{out_large}")
    print(f"  每行和: {out_large.sum(axis=1)}")
    print(f"  是否有 nan: {np.isnan(out_large).any()}")
    print(f"  是否有 inf: {np.isinf(out_large).any()}")
    
    # 测试4: 数值稳定性（小数值）
    print("\n[4] 数值稳定性测试（小数值）")
    x_small = np.array([[-1000, -1000, -1000],
                        [-1000, -999, -1000]], dtype=np.float32)
    out_small, _ = softmax(x_small)
    print(f"  输入: \n{x_small}")
    print(f"  输出: \n{out_small}")
    print(f"  每行和: {out_small.sum(axis=1)}")
    print(f"  是否有 nan: {np.isnan(out_small).any()}")
    print(f"  是否有 inf: {np.isinf(out_small).any()}")


def test_gradient():
    """梯度正确性验证"""
    print("\n" + "=" * 60)
    print("测试2: 梯度正确性验证")
    print("=" * 60)
    
    N, D = 3, 5
    x = np.random.randn(N, D).astype(np.float32)
    
    print(f"\n输入: {x.shape}")
    
    # 前向
    out, cache = softmax(x)
    loss = (out ** 2).sum()
    print(f"  Loss: {loss:.6f}")
    
    # 反向
    dloss = 2 * out
    dx = d_softmax(dloss, cache)
    print(f"  dx 形状: {dx.shape}")
    
    # 数值梯度检查
    print("\n数值梯度检查:")
    eps = 1e-6
    dx_num = np.zeros_like(x)
    
    for i in range(N):
        for j in range(D):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i, j] += eps
            x_minus[i, j] -= eps
            
            out_plus, _ = softmax(x_plus)
            out_minus, _ = softmax(x_minus)
            
            loss_plus = (out_plus ** 2).sum()
            loss_minus = (out_minus ** 2).sum()
            dx_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    max_diff = np.abs(dx - dx_num).max()
    print(f"  最大差异: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("  ✅ 梯度检查通过!")
    else:
        print("  ❌ 梯度检查失败!")


def test_softmax_cross_entropy():
    """测试 Softmax + 交叉熵损失"""
    print("\n" + "=" * 60)
    print("测试3: Softmax + 交叉熵损失")
    print("=" * 60)
    
    N, D = 4, 5
    x = np.random.randn(N, D).astype(np.float32)
    y = np.random.randint(0, D, size=N)  # 真实标签
    
    print(f"\n输入: {x.shape}")
    print(f"标签: {y}")
    
    # 前向
    loss, cache = softmax_cross_entropy(x, y)
    print(f"  Loss: {loss:.6f}")
    
    # 反向
    dx = d_softmax_cross_entropy(cache)
    print(f"  dx 形状: {dx.shape}")
    
    # 数值梯度检查
    print("\n数值梯度检查:")
    eps = 1e-6
    dx_num = np.zeros_like(x)
    
    for i in range(N):
        for j in range(D):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i, j] += eps
            x_minus[i, j] -= eps
            
            loss_plus, _ = softmax_cross_entropy(x_plus, y)
            loss_minus, _ = softmax_cross_entropy(x_minus, y)
            dx_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    max_diff = np.abs(dx - dx_num).max()
    print(f"  最大差异: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("  ✅ 梯度检查通过!")
    else:
        print("  ❌ 梯度检查失败!")


def test_softmax_cross_entropy_onehot():
    """测试 Softmax + 交叉熵（one-hot 标签）"""
    print("\n" + "=" * 60)
    print("测试4: Softmax + 交叉熵 (one-hot 标签)")
    print("=" * 60)
    
    N, D = 4, 5
    x = np.random.randn(N, D).astype(np.float32)
    y_onehot = np.eye(D)[np.random.randint(0, D, size=N)]
    
    print(f"\n输入: {x.shape}")
    print(f"标签 (one-hot): \n{y_onehot}")
    
    # 前向
    loss, cache = softmax_cross_entropy(x, y_onehot)
    print(f"  Loss: {loss:.6f}")
    
    # 反向
    dx = d_softmax_cross_entropy(cache)
    print(f"  dx 形状: {dx.shape}")
    
    # 数值梯度检查
    eps = 1e-6
    dx_num = np.zeros_like(x)
    
    for i in range(N):
        for j in range(D):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i, j] += eps
            x_minus[i, j] -= eps
            
            loss_plus, _ = softmax_cross_entropy(x_plus, y_onehot)
            loss_minus, _ = softmax_cross_entropy(x_minus, y_onehot)
            dx_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    max_diff = np.abs(dx - dx_num).max()
    print(f"  最大差异: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("  ✅ 梯度检查通过!")
    else:
        print("  ❌ 梯度检查失败!")


def test_properties():
    """测试 Softmax 的性质"""
    print("\n" + "=" * 60)
    print("测试5: Softmax 性质验证")
    print("=" * 60)
    
    N, D = 5, 10
    x = np.random.randn(N, D).astype(np.float32)
    out, _ = softmax(x)
    
    # 性质1: 所有输出 > 0
    print("\n[1] 正性: 所有输出 > 0")
    print(f"  最小值: {out.min():.8f}")
    print(f"  是否都 > 0: {np.all(out > 0)}")
    
    # 性质2: 每行和为 1
    print("\n[2] 归一化: 每行和为 1")
    row_sums = out.sum(axis=1)
    print(f"  每行和: {row_sums}")
    print(f"  是否都为 1: {np.allclose(row_sums, 1.0)}")
    
    # 性质3: 输出范围 [0, 1]
    print("\n[3] 范围: 输出在 [0, 1] 之间")
    print(f"  最小值: {out.min():.8f}")
    print(f"  最大值: {out.max():.8f}")
    print(f"  是否在 [0,1]: {np.all((out >= 0) & (out <= 1))}")
    
    # 性质4: 平移不变性
    print("\n[4] 平移不变性: softmax(x + c) = softmax(x)")
    c = 5.0
    x_shifted = x + c
    out_shifted, _ = softmax(x_shifted)
    diff = np.abs(out - out_shifted).max()
    print(f"  添加常数 {c} 后的差异: {diff:.8f}")
    print(f"  是否不变: {np.allclose(out, out_shifted)}")
    
    # 性质5: 单调性
    print("\n[5] 单调性: 输入越大，输出越大（相对）")
    x_sorted_idx = np.argsort(x, axis=1)
    out_sorted_idx = np.argsort(out, axis=1)
    
    # 检查排序是否一致（相对顺序）
    match_count = 0
    for i in range(N):
        if np.all(x_sorted_idx[i] == out_sorted_idx[i]):
            match_count += 1
    print(f"  排序完全匹配的行数: {match_count}/{N}")


def test_performance():
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试6: 性能测试")
    print("=" * 60)
    
    test_cases = [
        (1, 10, "N=1, D=10"),
        (64, 128, "N=64, D=128"),
        (128, 256, "N=128, D=256"),
        (256, 512, "N=256, D=512"),
        (512, 1024, "N=512, D=1024"),
    ]
    
    print("\n前向 + 反向性能 (10000次):")
    print("-" * 70)
    print(f"{'配置':<25} {'前向':<12} {'反向':<12} {'总计':<12}")
    print("-" * 70)
    
    for N, D, name in test_cases:
        x = np.random.randn(N, D).astype(np.float32)
        
        # 前向性能
        start = time.time()
        for _ in range(10000):
            out, cache = softmax(x)
        forward_time = time.time() - start
        
        # 反向性能
        dout = np.random.randn(*out.shape).astype(np.float32)
        start = time.time()
        for _ in range(10000):
            dx = d_softmax(dout, cache)
        backward_time = time.time() - start
        
        total_time = forward_time + backward_time
        
        print(f"{name:<25} {forward_time:.4f}s    {backward_time:.4f}s    {total_time:.4f}s")


def test_edge_cases():
    """边缘情况测试"""
    print("\n" + "=" * 60)
    print("测试7: 边缘情况测试")
    print("=" * 60)
    
    # 测试1: 所有值相等
    print("\n[1] 所有值相等")
    x = np.ones((3, 5), dtype=np.float32)
    out, _ = softmax(x)
    print(f"  输入: {x[0]}")
    print(f"  输出: {out[0]}")
    print(f"  期望: {1/5:.4f}")
    print(f"  是否符合: {np.allclose(out[0], 1/5)}")
    
    # 测试2: 单个样本
    print("\n[2] 单个样本 (N=1)")
    x = np.random.randn(1, 10).astype(np.float32)
    out, _ = softmax(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  是否和为1: {np.allclose(out.sum(), 1.0)}")
    
    # 测试3: D=1
    print("\n[3] D=1")
    x = np.random.randn(3, 1).astype(np.float32)
    out, _ = softmax(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  是否全为1: {np.allclose(out, 1.0)}")
    
    # 测试4: 非常大的 N
    print("\n[4] 大量样本 (N=1000, D=10)")
    x = np.random.randn(1000, 10).astype(np.float32)
    out, _ = softmax(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  是否每行和为1: {np.allclose(out.sum(axis=1), 1.0)}")
    
    # 测试5: 包含极端值
    print("\n[5] 包含极端值")
    x = np.array([[1000, -1000, 0]], dtype=np.float32)
    out, _ = softmax(x)
    print(f"  输入: {x}")
    print(f"  输出: {out}")
    print(f"  是否和为1: {np.allclose(out.sum(), 1.0)}")
    print(f"  是否有 nan: {np.isnan(out).any()}")
    print(f"  是否有 inf: {np.isinf(out).any()}")


def test_compare_with_scipy():
    """与 scipy.special.softmax 对比"""
    print("\n" + "=" * 60)
    print("测试8: 与 scipy.special.softmax 对比")
    print("=" * 60)
    
    try:
        from scipy.special import softmax as scipy_softmax
        
        N, D = 3, 5
        x = np.random.randn(N, D).astype(np.float32)
        
        # 我们的实现
        out_our, _ = softmax(x)
        
        # SciPy 实现
        out_scipy = scipy_softmax(x, axis=-1)
        
        diff = np.abs(out_our - out_scipy).max()
        print(f"\n输入: {x.shape}")
        print(f"最大差异: {diff:.8f}")
        print(f"{'✅ 结果一致!' if diff < 1e-6 else '❌ 存在差异!'}")
        
    except ImportError:
        print("  scipy 未安装，跳过对比测试")


if __name__ == "__main__":
    # 运行所有测试
    test_basic()
    test_gradient()
    test_softmax_cross_entropy()
    test_softmax_cross_entropy_onehot()
    test_properties()
    test_performance()
    test_edge_cases()
    test_compare_with_scipy()
    
    print("\n" + "=" * 60)
    print("所有测试完成! ✅")
    print("=" * 60)
