from funcs import conv, d_conv
import numpy as np

# 测试前向和反向
N, C, H, W = 2, 3, 28, 28
C_out, Hk, Wk = 16, 5, 5
stride, padding = 1, 2

# 随机数据
x = np.random.randn(N, C, H, W).astype(np.float32)
w = np.random.randn(C_out, C, Hk, Wk).astype(np.float32) * 0.01
b = np.random.randn(C_out).astype(np.float32)

print("输入形状:", x.shape)
print("权重形状:", w.shape)
print("偏置形状:", b.shape)

# 前向
out, x_pad, (H_out, W_out) = conv(x, w, b, stride, padding)
print("\n前向输出形状:", out.shape)
print("x_pad 形状:", x_pad.shape)
print("H_out, W_out:", H_out, W_out)

# 模拟上游梯度
dout = np.random.randn(*out.shape).astype(np.float32)

# 反向
dx, dw, db = d_conv(dout, x, w, stride, padding, x_pad, H_out, W_out)
print("\n反向梯度形状:")
print("dx 形状:", dx.shape)
print("dw 形状:", dw.shape)
print("db 形状:", db.shape)

# 验证梯度（数值梯度检查）
print("\n=== 梯度检查 ===")
eps = 1e-6

# 检查 dw
dw_num = np.zeros_like(w)
for i in range(min(3, C_out)):
    for j in range(min(3, C)):
        for kh in range(min(2, Hk)):
            for kw in range(min(2, Wk)):
                w_plus = w.copy()
                w_minus = w.copy()
                w_plus[i, j, kh, kw] += eps
                w_minus[i, j, kh, kw] -= eps
                out_plus, _, _ = conv(x, w_plus, b, stride, padding)
                out_minus, _, _ = conv(x, w_minus, b, stride, padding)
                loss_plus = (out_plus ** 2).sum()
                loss_minus = (out_minus ** 2).sum()
                dw_num[i, j, kh, kw] = (loss_plus - loss_minus) / (2 * eps)

print(f"dw 最大差异: {np.abs(dw - dw_num).max():.6f}")
print(f"dw 差异（前3个元素）: {np.abs(dw.flatten()[:10] - dw_num.flatten()[:10]).max():.6f}")

print("\n梯度统计:")
print(f"dx: min={dx.min():.4f}, max={dx.max():.4f}, mean={dx.mean():.4f}")
print(f"dw: min={dw.min():.4f}, max={dw.max():.4f}, mean={dw.mean():.4f}")
print(f"db: min={db.min():.4f}, max={db.max():.4f}, mean={db.mean():.4f}")
