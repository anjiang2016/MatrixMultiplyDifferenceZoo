import numpy as np
import time

from linear import linear, d_linear


def test_basic():
    """基础功能测试"""
    print("=" * 60)
    print("测试1: 基础功能测试")
    print("=" * 60)
    
    # 测试1: 标准线性层（带偏置）
    print("\n[1] 标准线性层（带偏置）")
    N, D_in, D_out = 2, 128, 64
    x = np.random.randn(N, D_in).astype(np.float32)
    w = np.random.randn(D_in, D_out).astype(np.float32) * 0.01
    b = np.random.randn(D_out).astype(np.float32)
    
    print(f"  输入: {x.shape}")
    print(f"  权重: {w.shape}")
    print(f"  偏置: {b.shape}")
    
    out, cache = linear(x, w, b)
    print(f"  输出: {out.shape}")
    print(f"  期望: ({N}, {D_out})")
    
    # 反向
    dout = np.random.randn(*out.shape).astype(np.float32)
    dx, dw, db = d_linear(dout, cache)
    
    print(f"\n  梯度形状:")
    print(f"    dx: {dx.shape}")
    print(f"    dw: {dw.shape}")
    print(f"    db: {db.shape}")
    
    # 测试2: 不带偏置
    print("\n[2] 线性层（不带偏置）")
    out2, cache2 = linear(x, w, b=None)
    print(f"  输出: {out2.shape}")
    
    dout2 = np.random.randn(*out2.shape).astype(np.float32)
    dx2, dw2, db2 = d_linear(dout2, cache2)
    
    print(f"  梯度形状:")
    print(f"    dx: {dx2.shape}")
    print(f"    dw: {dw2.shape}")
    print(f"    db: {db2}")
    
    # 测试3: 批量大小 1
    print("\n[3] 批量大小 1")
    x3 = np.random.randn(1, D_in).astype(np.float32)
    out3, cache3 = linear(x3, w, b)
    print(f"  输入: {x3.shape} -> 输出: {out3.shape}")
    
    dout3 = np.random.randn(*out3.shape).astype(np.float32)
    dx3, dw3, db3 = d_linear(dout3, cache3)
    print(f"  梯度: dx {dx3.shape}, dw {dw3.shape}, db {db3.shape}")
    
    # 测试4: 大批量
    print("\n[4] 大批量 (N=100)")
    x4 = np.random.randn(100, D_in).astype(np.float32)
    out4, cache4 = linear(x4, w, b)
    print(f"  输入: {x4.shape} -> 输出: {out4.shape}")
    
    dout4 = np.random.randn(*out4.shape).astype(np.float32)
    dx4, dw4, db4 = d_linear(dout4, cache4)
    print(f"  梯度: dx {dx4.shape}, dw {dw4.shape}, db {db4.shape}")


def test_gradient():
    """梯度正确性验证"""
    print("\n" + "=" * 60)
    print("测试2: 梯度正确性验证")
    print("=" * 60)
    
    N, D_in, D_out = 3, 5, 4
    x = np.random.randn(N, D_in).astype(np.float32)
    w = np.random.randn(D_in, D_out).astype(np.float32) * 0.01
    b = np.random.randn(D_out).astype(np.float32)
    
    print(f"\n网络: 输入 {x.shape} -> Linear -> 输出 {D_out}")
    
    # 前向
    out, cache = linear(x, w, b)
    loss = (out ** 2).sum()
    print(f"  Loss: {loss:.6f}")
    
    # 反向
    dloss = 2 * out  # dloss/dout
    dx, dw, db = d_linear(dloss, cache)
    
    print(f"\n  梯度形状: dx {dx.shape}, dw {dw.shape}, db {db.shape}")
    
    # ========== 数值梯度检查 ==========
    print("\n数值梯度检查:")
    eps = 1e-6
    
    # 检查 dx
    print("\n[1] 检查 dx:")
    dx_num = np.zeros_like(x)
    for i in range(N):
        for j in range(D_in):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i, j] += eps
            x_minus[i, j] -= eps
            
            out_plus, _ = linear(x_plus, w, b)
            out_minus, _ = linear(x_minus, w, b)
            
            loss_plus = (out_plus ** 2).sum()
            loss_minus = (out_minus ** 2).sum()
            dx_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    dx_diff = np.abs(dx - dx_num).max()
    print(f"    dx 最大差异: {dx_diff:.8f}")
    print(f"    {'✅ 通过' if dx_diff < 1e-5 else '❌ 失败'}")
    
    # 检查 dw
    print("\n[2] 检查 dw:")
    dw_num = np.zeros_like(w)
    for i in range(D_in):
        for j in range(D_out):
            w_plus = w.copy()
            w_minus = w.copy()
            w_plus[i, j] += eps
            w_minus[i, j] -= eps
            
            out_plus, _ = linear(x, w_plus, b)
            out_minus, _ = linear(x, w_minus, b)
            
            loss_plus = (out_plus ** 2).sum()
            loss_minus = (out_minus ** 2).sum()
            dw_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    dw_diff = np.abs(dw - dw_num).max()
    print(f"    dw 最大差异: {dw_diff:.8f}")
    print(f"    {'✅ 通过' if dw_diff < 1e-5 else '❌ 失败'}")
    
    # 检查 db
    print("\n[3] 检查 db:")
    db_num = np.zeros_like(b)
    for j in range(D_out):
        b_plus = b.copy()
        b_minus = b.copy()
        b_plus[j] += eps
        b_minus[j] -= eps
        
        out_plus, _ = linear(x, w, b_plus)
        out_minus, _ = linear(x, w, b_minus)
        
        loss_plus = (out_plus ** 2).sum()
        loss_minus = (out_minus ** 2).sum()
        db_num[j] = (loss_plus - loss_minus) / (2 * eps)
    
    db_diff = np.abs(db - db_num).max()
    print(f"    db 最大差异: {db_diff:.8f}")
    print(f"    {'✅ 通过' if db_diff < 1e-5 else '❌ 失败'}")


def test_performance():
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试3: 性能测试")
    print("=" * 60)
    
    test_cases = [
        (1, 128, 64, "N=1, D_in=128, D_out=64"),
        (64, 256, 128, "N=64, D_in=256, D_out=128"),
        (128, 512, 256, "N=128, D_in=512, D_out=256"),
        (256, 1024, 512, "N=256, D_in=1024, D_out=512"),
    ]
    
    print("\n前向 + 反向性能 (10000次):")
    print("-" * 70)
    print(f"{'配置':<35} {'前向':<12} {'反向':<12} {'总计':<12}")
    print("-" * 70)
    
    for N, D_in, D_out, name in test_cases:
        x = np.random.randn(N, D_in).astype(np.float32)
        w = np.random.randn(D_in, D_out).astype(np.float32) * 0.01
        b = np.random.randn(D_out).astype(np.float32)
        
        # 前向性能
        start = time.time()
        for _ in range(10000):
            out, cache = linear(x, w, b)
        forward_time = time.time() - start
        
        # 反向性能
        dout = np.random.randn(*out.shape).astype(np.float32)
        start = time.time()
        for _ in range(10000):
            dx, dw, db = d_linear(dout, cache)
        backward_time = time.time() - start
        
        total_time = forward_time + backward_time
        
        print(f"{name:<35} {forward_time:.4f}s    {backward_time:.4f}s    {total_time:.4f}s")


def test_compare_with_numpy():
    """与 NumPy 原生实现对比"""
    print("\n" + "=" * 60)
    print("测试4: 与 NumPy 对比")
    print("=" * 60)
    
    N, D_in, D_out = 4, 10, 6
    x = np.random.randn(N, D_in).astype(np.float32)
    w = np.random.randn(D_in, D_out).astype(np.float32) * 0.01
    b = np.random.randn(D_out).astype(np.float32)
    
    print(f"\n输入: {x.shape}, 权重: {w.shape}, 偏置: {b.shape}")
    
    # 我们的实现
    out_our, cache = linear(x, w, b)
    dout = np.random.randn(*out_our.shape).astype(np.float32)
    dx_our, dw_our, db_our = d_linear(dout, cache)
    
    # NumPy 原生
    out_np = x @ w + b
    dx_np = dout @ w.T
    dw_np = x.T @ dout
    db_np = dout.sum(axis=0)
    
    # 对比
    print("\n对比结果:")
    print(f"  out 差异: {np.abs(out_our - out_np).max():.8f}")
    print(f"  dx 差异: {np.abs(dx_our - dx_np).max():.8f}")
    print(f"  dw 差异: {np.abs(dw_our - dw_np).max():.8f}")
    print(f"  db 差异: {np.abs(db_our - db_np).max():.8f}")
    
    all_match = (
        np.allclose(out_our, out_np) and
        np.allclose(dx_our, dx_np) and
        np.allclose(dw_our, dw_np) and
        np.allclose(db_our, db_np)
    )
    
    print(f"\n{'✅ 所有结果一致!' if all_match else '❌ 存在差异!'}")


def test_edge_cases():
    """边缘情况测试"""
    print("\n" + "=" * 60)
    print("测试5: 边缘情况测试")
    print("=" * 60)
    
    # 测试1: D_out = 1
    print("\n[1] D_out = 1")
    x = np.random.randn(4, 10).astype(np.float32)
    w = np.random.randn(10, 1).astype(np.float32)
    b = np.random.randn(1).astype(np.float32)
    
    out, cache = linear(x, w, b)
    print(f"  输入: {x.shape} -> 输出: {out.shape}")
    
    dout = np.random.randn(*out.shape).astype(np.float32)
    dx, dw, db = d_linear(dout, cache)
    print(f"  梯度: dx {dx.shape}, dw {dw.shape}, db {db.shape}")
    
    # 测试2: D_in = 1
    print("\n[2] D_in = 1")
    x = np.random.randn(4, 1).astype(np.float32)
    w = np.random.randn(1, 5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    
    out, cache = linear(x, w, b)
    print(f"  输入: {x.shape} -> 输出: {out.shape}")
    
    dout = np.random.randn(*out.shape).astype(np.float32)
    dx, dw, db = d_linear(dout, cache)
    print(f"  梯度: dx {dx.shape}, dw {dw.shape}, db {db.shape}")
    
    # 测试3: 零输入
    print("\n[3] 零输入")
    x = np.zeros((3, 5), dtype=np.float32)
    w = np.random.randn(5, 4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    
    out, cache = linear(x, w, b)
    print(f"  输入: {x.shape} -> 输出: {out.shape}")
    print(f"  输出是否等于偏置: {np.allclose(out, b)}")
    
    dout = np.random.randn(*out.shape).astype(np.float32)
    dx, dw, db = d_linear(dout, cache)
    print(f"  零输入的 dx 是否为零: {np.all(dx == 0)}")
    
    # 测试4: 大数值
    print("\n[4] 大数值")
    x = np.random.randn(3, 10).astype(np.float32) * 100
    w = np.random.randn(10, 5).astype(np.float32) * 100
    b = np.random.randn(5).astype(np.float32) * 100
    
    out, cache = linear(x, w, b)
    print(f"  输入范围: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  输出范围: [{out.min():.2f}, {out.max():.2f}]")
    
    dout = np.random.randn(*out.shape).astype(np.float32)
    dx, dw, db = d_linear(dout, cache)
    print(f"  dx 范围: [{dx.min():.2f}, {dx.max():.2f}]")
    print(f"  dw 范围: [{dw.min():.2f}, {dw.max():.2f}]")


def test_consistency():
    """一致性测试：验证链式法则"""
    print("\n" + "=" * 60)
    print("测试6: 链式法则一致性测试")
    print("=" * 60)
    
    # 构建两层网络: x -> Linear1 -> Linear2 -> loss
    N, D1, D2, D3 = 4, 8, 16, 10
    
    x = np.random.randn(N, D1).astype(np.float32)
    w1 = np.random.randn(D1, D2).astype(np.float32) * 0.01
    w2 = np.random.randn(D2, D3).astype(np.float32) * 0.01
    b1 = np.random.randn(D2).astype(np.float32)
    b2 = np.random.randn(D3).astype(np.float32)
    
    print(f"\n网络: {D1} -> Linear1 -> {D2} -> Linear2 -> {D3}")
    
    # 前向
    out1, cache1 = linear(x, w1, b1)
    out2, cache2 = linear(out1, w2, b2)
    loss = (out2 ** 2).sum()
    
    print(f"  Loss: {loss:.6f}")
    
    # 反向
    dloss = 2 * out2
    
    # Layer2 反向
    dout1, dw2, db2 = d_linear(dloss, cache2)
    
    # Layer1 反向
    dx, dw1, db1 = d_linear(dout1, cache1)
    
    print(f"\n梯度形状:")
    print(f"  dx: {dx.shape}")
    print(f"  dw1: {dw1.shape}, db1: {db1.shape}")
    print(f"  dw2: {dw2.shape}, db2: {db2.shape}")
    
    # 验证 dw1 是否正确（使用数值梯度）
    print("\n验证 dw1 (数值梯度):")
    eps = 1e-6
    dw1_num = np.zeros_like(w1)
    
    for i in range(min(3, D1)):
        for j in range(min(3, D2)):
            w1_plus = w1.copy()
            w1_minus = w1.copy()
            w1_plus[i, j] += eps
            w1_minus[i, j] -= eps
            
            out1_plus, _ = linear(x, w1_plus, b1)
            out2_plus, _ = linear(out1_plus, w2, b2)
            loss_plus = (out2_plus ** 2).sum()
            
            out1_minus, _ = linear(x, w1_minus, b1)
            out2_minus, _ = linear(out1_minus, w2, b2)
            loss_minus = (out2_minus ** 2).sum()
            
            dw1_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    dw1_diff = np.abs(dw1[:3, :3] - dw1_num[:3, :3]).max()
    print(f"  dw1 最大差异: {dw1_diff:.8f}")
    print(f"  {'✅ 通过' if dw1_diff < 1e-5 else '❌ 失败'}")


if __name__ == "__main__":
    # 运行所有测试
    test_basic()
    test_gradient()
    test_performance()
    test_compare_with_numpy()
    test_edge_cases()
    test_consistency()
    
    print("\n" + "=" * 60)
    print("所有测试完成! ✅")
    print("=" * 60)
