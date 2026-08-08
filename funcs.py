# ============================================================
# 尝试导入 C 扩展（如果已编译）
# ============================================================
try:
    import convmodule
    C_EXT_AVAILABLE = True
    C_EXT_DCONV_AVAILABLE = hasattr(convmodule, 'd_conv_c')
except ImportError:
    C_EXT_AVAILABLE = False
    C_EXT_DCONV_AVAILABLE = False
    print("⚠️ C 扩展未找到，使用纯 NumPy 实现（速度较慢）")
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os

# ============================================================
# 后端选择（通过环境变量 BACKEND 控制）
# ============================================================
# 可选值: 'numpy', 'c'
# 默认: 'c'
# 使用方式: BACKEND=numpy python train.py
BACKEND = os.environ.get('BACKEND', 'numpy')
#C_EXT_AVAILABLE = False
print(f'BACKEND = {BACKEND}')
# ============================================================
# 卷积前向（自动选择 C 或 NumPy）
# ============================================================
def conv(x, w, b=None, stride=1, padding=0):
    N, C, H, W = x.shape
    C_out, _, Hk, Wk = w.shape

    H_out = (H + 2*padding - Hk) // stride + 1
    W_out = (W + 2*padding - Wk) // stride + 1

    if C_EXT_AVAILABLE:
        x_contig = np.ascontiguousarray(x, dtype=np.float32)
        w_contig = np.ascontiguousarray(w, dtype=np.float32)
        out = convmodule.conv_c(x_contig, w_contig, stride, padding)
        if b is not None:
            out += b.reshape(1, -1, 1, 1)
        x_pad = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
        cache = (x, w, b, stride, padding, x_pad, H_out, W_out)
        # ✅ 返回三个值：out, cache, x_pad
        #return out, cache, x_pad
        return out, x_pad,(H_out,W_out)

    if BACKEND == 'numpy':
        # ---------- 纯 NumPy 实现 ----------
        x_pad = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
        shape = (N, C, H_out, W_out, Hk, Wk)
        strides = (
            x_pad.strides[0],
            x_pad.strides[1],
            x_pad.strides[2] * stride,
            x_pad.strides[3] * stride,
            x_pad.strides[2],
            x_pad.strides[3]
        )
        cols = as_strided(x_pad, shape=shape, strides=strides)
        cols = cols.reshape(N, C, H_out * W_out, Hk * Wk)
        cols = cols.transpose(0, 2, 1, 3).reshape(N * H_out * W_out, C * Hk * Wk)
        w_flat = w.reshape(C_out, C * Hk * Wk)
        out = cols @ w_flat.T
        out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)
        if b is not None:
            out += b.reshape(1, -1, 1, 1)
        cache = (x, w, b, stride, padding, x_pad, H_out, W_out)
        # ✅ 返回三个值：out, cache, x_pad
        #return out, cache, x_pad
        return out, x_pad,(H_out,W_out)


# ============================================================
# 卷积反向（自动选择 C 或 NumPy）
# ============================================================
def d_conv(dout, x, w, stride=1, padding=0, x_pad=None, H_out=None, W_out=None):
    """
    卷积反向传播（自动选择 C 扩展或纯 NumPy）
    """
    N, C, H, W = x.shape
    C_out, _, Hk, Wk = w.shape

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    # 如果未传入 x_pad，则计算（C 扩展也需要）
    if x_pad is None:
        x_pad = np.pad(x, ((0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # 计算 H_out, W_out（如果未传入）
    if H_out is None or W_out is None:
        H_out = (H + 2*pad_h - Hk) // stride_h + 1
        W_out = (W + 2*pad_w - Wk) // stride_w + 1

    # ---------- 优先使用 C 扩展 ----------
    if C_EXT_DCONV_AVAILABLE:
        # 确保数据是 float32 且内存连续
        dout_contig = np.ascontiguousarray(dout, dtype=np.float32)
        x_contig = np.ascontiguousarray(x, dtype=np.float32)
        w_contig = np.ascontiguousarray(w, dtype=np.float32)
        x_pad_contig = np.ascontiguousarray(x_pad, dtype=np.float32)

        # 调用 C 函数（d_conv_c 签名：d_conv_c(dout, x, w, stride, padding, x_pad)）
        dx, dw, db = convmodule.d_conv_c(
            dout_contig, x_contig, w_contig,
            stride, padding, x_pad_contig
        )
        return dx, dw, db
    if BACKEND == 'numpy':
        # ---------- 回退到纯 NumPy 实现 ----------
        # （如果 C 扩展不可用，使用纯 NumPy 版本）
        # db
        db = dout.sum(axis=(0, 2, 3))

        # dw
        shape = (N, C, H_out, W_out, Hk, Wk)
        strides = (
            x_pad.strides[0],
            x_pad.strides[1],
            x_pad.strides[2] * stride_h,
            x_pad.strides[3] * stride_w,
            x_pad.strides[2],
            x_pad.strides[3]
        )
        cols = as_strided(x_pad, shape=shape, strides=strides)
        cols = cols.reshape(N, C, H_out * W_out, Hk * Wk)
        cols = cols.transpose(0, 2, 1, 3).reshape(N * H_out * W_out, C * Hk * Wk)

        dout_flat = dout.transpose(0, 2, 3, 1).reshape(N * H_out * W_out, C_out)
        dw = dout_flat.T @ cols
        dw = dw.reshape(C_out, C, Hk, Wk)

        # dx
        w_flat = w.reshape(C_out, C * Hk * Wk).T
        dcols = dout_flat @ w_flat.T
        dcols = dcols.reshape(N, H_out, W_out, C, Hk, Wk).transpose(0, 3, 1, 2, 4, 5)

        dx_pad = np.zeros((N, C, H + 2*pad_h, W + 2*pad_w), dtype=dcols.dtype)
        for h_off in range(Hk):
            for w_off in range(Wk):
                dcol_slice = dcols[:, :, :, :, h_off, w_off]
                h_start = h_off
                h_end = h_off + H_out * stride_h
                w_start = w_off
                w_end = w_off + W_out * stride_w
                h_slice = slice(h_start, h_end, stride_h)
                w_slice = slice(w_start, w_end, stride_w)
                dx_pad[:, :, h_slice, w_slice] += dcol_slice

        if pad_h > 0 or pad_w > 0:
            dx = dx_pad[:, :, pad_h:-pad_h, pad_w:-pad_w]
        else:
            dx = dx_pad

        return dx, dw, db

def sigmoid(x):
    """
    Sigmoid 激活函数
    
    Args:
        x: 输入，可以是标量、向量或矩阵
    
    Returns:
        out: sigmoid(x) = 1 / (1 + exp(-x))
    
    Notes:
        - 数值稳定版本，防止 exp 溢出
        - 对于 x >= 0: 使用 1 / (1 + exp(-x))
        - 对于 x < 0: 使用 exp(x) / (1 + exp(x))
    """
    x=np.clip(x,-20,20)
    # 数值稳定版本
    out = np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )
    return out


def d_sigmoid(x, y=None):
    """
    Sigmoid 的导数
    
    Args:
        x: 输入值（前向传播的输入）
        y: 可选，sigmoid(x) 的输出值
    
    Returns:
        grad: sigmoid'(x)
    """
    if y is not None:
        return y * (1 - y)
    else:
        s = sigmoid(x)
        return s * (1 - s)

import numpy as np
from numpy.lib.stride_tricks import as_strided

def avgpool(x, kernel_size, stride=None, padding=0):
    """
    平均池化前向传播
    
    Args:
        x: (N, C, H, W) 输入
        kernel_size: int 或 tuple (Hk, Wk)
        stride: int 或 tuple，默认等于 kernel_size
        padding: int 或 tuple
    
    Returns:
        out: (N, C, H_out, W_out) 池化后的输出
        cache: (x_shape, kernel_size, stride, padding) 用于反向传播
    """
    N, C, H, W = x.shape
    
    if isinstance(kernel_size, int):
        Hk = Wk = kernel_size
    else:
        Hk, Wk = kernel_size
    
    if stride is None:
        stride_h = stride_w = Hk
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    # Padding
    x_pad = np.pad(x, ((0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # 输出尺寸
    H_out = (H + 2*pad_h - Hk) // stride_h + 1
    W_out = (W + 2*pad_w - Wk) // stride_w + 1
    
    # im2col 提取块
    shape = (N, C, H_out, W_out, Hk, Wk)
    strides = (
        x_pad.strides[0],
        x_pad.strides[1],
        x_pad.strides[2] * stride_h,
        x_pad.strides[3] * stride_w,
        x_pad.strides[2],
        x_pad.strides[3]
    )
    cols = as_strided(x_pad, shape=shape, strides=strides)
    
    # 平均池化：对 Hk*Wk 维度求平均
    out = cols.mean(axis=(4, 5))  # (N, C, H_out, W_out)
    
    # 缓存用于反向传播
    cache = (x.shape, kernel_size, stride, padding, x_pad, H_out, W_out)
    
    return out, cache


def d_avgpool(dout, cache):
    """
    平均池化反向传播
    
    Args:
        dout: (N, C, H_out, W_out) 上游梯度
        cache: 前向传播缓存的 (x_shape, kernel_size, stride, padding, x_pad, H_out, W_out)
    
    Returns:
        dx: (N, C, H, W) 输入梯度
    """
    x_shape, kernel_size, stride, padding, x_pad, H_out, W_out = cache
    
    N, C, H, W = x_shape
    
    if isinstance(kernel_size, int):
        Hk = Wk = kernel_size
    else:
        Hk, Wk = kernel_size
    
    if stride is None:
        stride_h = stride_w = Hk
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    # 平均池化的反向传播：梯度均分到每个像素
    # 每个输出元素对应 Hk*Wk 个输入元素，每个分到 1/(Hk*Wk) 的梯度
    pool_size = Hk * Wk
    dout_expanded = dout / pool_size  # 梯度均分
    
    # 初始化 padding 后的梯度
    dx_pad = np.zeros((N, C, H + 2*pad_h, W + 2*pad_w), dtype=dout.dtype)
    
    # 将梯度分配到对应的输入位置
    # 方法1: 循环（清晰但较慢）
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride_h
                    h_end = h_start + Hk
                    w_start = w * stride_w
                    w_end = w_start + Wk
                    dx_pad[n, c, h_start:h_end, w_start:w_end] += dout_expanded[n, c, h, w]
    
    # 去掉 padding
    if pad_h > 0 or pad_w > 0:
        dx = dx_pad[:, :, pad_h:-pad_h, pad_w:-pad_w]
    else:
        dx = dx_pad
    
    return dx

def d_avgpool_f(dout, cache):
    """
    平均池化的反向传播（向量化版本）
    cache 应为 (input_shape, kernel_size, stride, padding)
    input_shape: (B, C, H_in, W_in)
    """
    x_shape, k, s, pad,_,_,_ = cache
    B, C, H_out, W_out = dout.shape
    H_in, W_in = x_shape[2], x_shape[3]
    
    # 初始化梯度数组
    dx = np.zeros((B, C, H_in, W_in), dtype=dout.dtype)
    
    # 1. 生成输出位置网格
    h_out = np.arange(H_out)
    w_out = np.arange(W_out)
    
    # 2. 计算每个输出窗口的起始位置（考虑 padding）
    h_start = h_out[:, None] * s - pad   # (H_out, 1)
    w_start = w_out[None, :] * s - pad   # (1, W_out)
    
    # 3. 生成窗口内偏移
    h_off = np.arange(k)
    w_off = np.arange(k)
    
    # 4. 生成所有输入索引 (H_out, W_out, k, k)
    h_idx = h_start[..., None, None] + h_off[None, None, :, None]  # (H_out, W_out, k, 1)
    w_idx = w_start[..., None, None] + w_off[None, None, None, :]  # (H_out, W_out, 1, k)
    h_idx = np.broadcast_to(h_idx, (H_out, W_out, k, k))
    w_idx = np.broadcast_to(w_idx, (H_out, W_out, k, k))
    
    # 5. 筛选有效位置（边界裁剪）
    valid = (h_idx >= 0) & (h_idx < H_in) & (w_idx >= 0) & (w_idx < W_in)
    valid_flat = valid.reshape(-1)
    
    # 6. 展平有效索引
    h_flat = h_idx.reshape(-1)[valid_flat]
    w_flat = w_idx.reshape(-1)[valid_flat]
    
    # 7. 对应的输出位置线性索引（用于取出 dout 值）
    out_idx = np.arange(H_out * W_out).reshape(H_out, W_out, 1, 1)
    out_idx = np.broadcast_to(out_idx, (H_out, W_out, k, k)).reshape(-1)[valid_flat]
    
    # 8. 对每个 batch 和 channel 进行累加（避免超大索引）
    for b in range(B):
        for c in range(C):
            # 取出该层对应的输出梯度（展平）
            dout_bc = dout[b, c].reshape(-1)  # (H_out*W_out,)
            # 每个窗口的有效像素数（用于平均）——这里固定为 k*k，因为前向平均时包括填充
            vals = dout_bc[out_idx] / (k * k)
            # 累加到 dx 的对应位置
            np.add.at(dx[b, c], (h_flat, w_flat), vals)
    
    return dx

def avgpool_fast(x, kernel_size, stride=None, padding=0):
    """
    使用矩阵运算加速的平均池化
    """
    N, C, H, W = x.shape
    
    if isinstance(kernel_size, int):
        Hk = Wk = kernel_size
    else:
        Hk, Wk = kernel_size
    
    if stride is None:
        stride_h = stride_w = Hk
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    x_pad = np.pad(x, ((0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    H_out = (H + 2*pad_h - Hk) // stride_h + 1
    W_out = (W + 2*pad_w - Wk) // stride_w + 1
    
    # 使用 view + mean
    shape = (N, C, H_out, W_out, Hk, Wk)
    strides = (
        x_pad.strides[0],
        x_pad.strides[1],
        x_pad.strides[2] * stride_h,
        x_pad.strides[3] * stride_w,
        x_pad.strides[2],
        x_pad.strides[3]
    )
    cols = as_strided(x_pad, shape=shape, strides=strides)
    out = cols.mean(axis=(4, 5))
    
    return out


def d_avgpool_fast(dout, x_shape, kernel_size, stride=None, padding=0):
    """
    使用高级索引加速的反向传播
    """
    N, C, H, W = x_shape
    
    if isinstance(kernel_size, int):
        Hk = Wk = kernel_size
    else:
        Hk, Wk = kernel_size
    
    if stride is None:
        stride_h = stride_w = Hk
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    H_out = (H + 2*pad_h - Hk) // stride_h + 1
    W_out = (W + 2*pad_w - Wk) // stride_w + 1
    
    pool_size = Hk * Wk
    dout_scaled = dout / pool_size
    
    dx_pad = np.zeros((N, C, H + 2*pad_h, W + 2*pad_w), dtype=dout.dtype)
    
    # 使用广播和索引加速
    for h in range(Hk):
        for w in range(Wk):
            h_indices = np.arange(h, h + H_out * stride_h, stride_h)
            w_indices = np.arange(w, w + W_out * stride_w, stride_w)
            dx_pad[:, :, h_indices[:, None], w_indices] += dout_scaled
    
    if pad_h > 0 or pad_w > 0:
        dx = dx_pad[:, :, pad_h:-pad_h, pad_w:-pad_w]
    else:
        dx = dx_pad
    
    return dx
	# ============ 更完整的版本（支持多种输入格式）============

def flatten(x):
    """
    增强版 Flatten，支持多种输入格式

    Args:
        x: 任意维度的输入

    Returns:
        out: 保留 batch 维度的展平结果
        cache: (x_shape, x_dtype) 缓存信息
    """
    if x.ndim == 1:
        # 单样本，无 batch 维度
        out = x.reshape(1, -1)
        cache = (x.shape, '1d')
    elif x.ndim == 2:
        # 已有 batch 维度，直接展平
        out = x.reshape(x.shape[0], -1)
        cache = (x.shape, '2d')
    else:
        # 多维，保留 batch 维度
        out = x.reshape(x.shape[0], -1)
        cache = (x.shape, 'nd')

    return out, cache


def d_flatten(dout, cache):
    """
    增强版 D_Flatten，支持多种输入格式
    """
    shape, mode = cache

    if mode == '1d':
        # 1D 输入，输出也是 1D
        dx = dout.reshape(shape)
    else:
        # 2D 或 ND 输入
        dx = dout.reshape(shape)

    return dx

def linear(x, w, b=None):
    """
    全连接层前向传播

    Args:
        x: (N, D_in) 输入
        w: (D_in, D_out) 权重
        b: (D_out,) 偏置，可选

    Returns:
        out: (N, D_out) 输出
        cache: (x, w, b) 用于反向传播
    """
    # 矩阵乘法
    out = x @ w  # (N, D_out)

    if b is not None:
        out += b

    cache = (x, w, b)
    return out, cache


def d_linear(dout, cache):
    """
    全连接层反向传播

    Args:
        dout: (N, D_out) 上游梯度
        cache: 前向传播缓存的 (x, w, b)

    Returns:
        dx: (N, D_in) 输入梯度
        dw: (D_in, D_out) 权重梯度
        db: (D_out,) 偏置梯度（如果有）
    """
    x, w, b = cache

    # 计算 dx: dout @ w.T
    dx = dout @ w.T  # (N, D_in)

    # 计算 dw: x.T @ dout
    dw = x.T @ dout  # (D_in, D_out)

    # 计算 db: dout 在 batch 维度上求和
    if b is not None:
        db = dout.sum(axis=0)  # (D_out,)
    else:
        db = None

    return dx, dw, db
import numpy as np

def softmax(x, axis=-1):
    """
    Softmax 激活函数
    
    Args:
        x: 输入，可以是 (N, D) 或 (N, C, H, W) 等任意维度
        axis: 在哪个维度上做 softmax，默认 -1（最后一个维度）
    
    Returns:
        out: softmax 输出，形状与 x 相同
        cache: 缓存 softmax 输出，用于反向传播
    """
    # 数值稳定版本：减去最大值
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    
    # 计算 exp
    exp_x = np.exp(x_shifted)
    
    # 计算 softmax
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    out = exp_x / sum_exp
    
    cache = out  # 缓存输出用于反向传播
    return out, cache


def d_softmax(dout, cache):
    """
    Softmax 反向传播
    
    Args:
        dout: (N, D) 上游梯度
        cache: 前向传播缓存的 softmax 输出
    
    Returns:
        dx: (N, D) 输入梯度
    
    Notes:
        如果配合交叉熵损失使用，可以直接返回 dout - softmax
        但这里是通用的 softmax 导数实现
    """
    s = cache  # softmax 输出 (N, D)
    
    # 计算 Jacobian 矩阵并乘以梯度
    # dx_i = s_i * (dout_i - sum(dout_j * s_j))
    # 这个公式是 softmax 导数的高效实现
    
    # 计算 s * dout 的和
    sum_s_dout = np.sum(s * dout, axis=-1, keepdims=True)  # (N, 1)
    
    # 计算梯度
    dx = s * (dout - sum_s_dout)  # (N, D)
    
    return dx


def softmax_cross_entropy(x, y, axis=-1):
    """
    Softmax + 交叉熵损失（联合前向）
    
    Args:
        x: (N, D) 输入 logits
        y: (N,) 真实标签（整数索引），或 (N, D) one-hot 编码
        axis: softmax 维度
    
    Returns:
        loss: 标量，平均交叉熵损失
        cache: (softmax_out, y, N) 用于反向传播
    """
    # 计算 softmax
    s, _ = softmax(x, axis)
    N = s.shape[0]
    
    # 计算交叉熵损失
    if y.ndim == 1:
        # y 是整数索引
        log_likelihood = -np.log(s[np.arange(N), y] + 1e-12)
    else:
        # y 是 one-hot
        log_likelihood = -np.sum(y * np.log(s + 1e-12), axis=1)
    
    loss = np.mean(log_likelihood)
    
    cache = (s, y, N)
    return loss, cache


def d_softmax_cross_entropy(cache):
    """
    Softmax + 交叉熵损失的反向传播
    
    Args:
        cache: 前向传播缓存的 (s, y, N)
    
    Returns:
        dx: (N, D) 输入梯度
    
    Notes:
        这是最常用的组合，梯度为 (softmax - one_hot) / N
    """
    s, y, N = cache
    
    # 如果 y 是整数索引，转为 one-hot
    if y.ndim == 1:
        D = s.shape[1]
        y_one_hot = np.zeros_like(s)
        y_one_hot[np.arange(N), y] = 1
    else:
        y_one_hot = y
    
    # 梯度: (softmax - one_hot) / N
    dx = (s - y_one_hot) / N
    
    return dx


# ============ 通用的 softmax 导数实现 ============

def softmax_general(x):
    """
    通用 softmax 前向（不缓存）
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def d_softmax_general(dout, s):
    """
    通用 softmax 反向（传入 softmax 输出）
    """
    # s: softmax 输出, dout: 上游梯度
    sum_s_dout = np.sum(s * dout, axis=-1, keepdims=True)
    return s * (dout - sum_s_dout)


def softmax_stable(x, axis=-1):
    """
    数值稳定的 softmax（带 axis 参数）
    """
    # 减去最大值
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp
def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def d_relu(x, y=None):
    """ReLU 导数"""
    if y is not None:
        return (y > 0).astype(np.float32)
    else:
        return (x > 0).astype(np.float32)
def silu(x):
    """SiLU (Swish) 激活函数: x * sigmoid(x)"""
    return x * sigmoid(x)

def d_silu(dout, x):
    """SiLU 导数: dout * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))"""
    sig = sigmoid(x)
    # 简化版导数: sig * (1 + x * (1 - sig))
    grad = sig * (1 + x * (1 - sig))
    return dout * grad
def dropout(x, keep_prob=0.5, training=True):
    """
    Dropout 前向传播

    Args:
        x: (N, D) 输入
        keep_prob: float, 保留神经元的概率
        training: bool, 是否训练模式（训练时丢弃，测试时保留所有）

    Returns:
        out: 输出
        cache: 掩码和 keep_prob，用于反向传播
    """
    if not training:
        # 测试阶段：保留所有神经元，输出不变
        return x, None

    # 训练阶段：随机丢弃神经元
    # 生成与 x 相同形状的随机掩码，保留概率为 keep_prob
    mask = np.random.rand(*x.shape) < keep_prob  # bool 数组
    out = x * mask  # 丢弃的神经元输出为 0

    # Inverted Dropout：除以 keep_prob 保持期望不变
    out = out / keep_prob

    cache = (mask, keep_prob)
    return out, cache


def d_dropout(dout, cache):
    """
    Dropout 反向传播

    Args:
        dout: (N, D) 上游梯度
        cache: (mask, keep_prob) 前向传播缓存的掩码和概率

    Returns:
        dx: (N, D) 输入梯度
    """
    if cache is None:
        # 测试阶段没有缓存，梯度直接通过
        return dout

    mask, keep_prob = cache
    # 梯度只通过训练时保留的神经元
    dx = dout * mask / keep_prob
    return dx
# ========== 内部实现 layer_norm 和 d_layer_norm ==========
def layer_norm(x, gamma, beta, eps=1e-5, cache=None):
    """
    前向：Layer Normalization
    返回 out, cache
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    cache = (x, mean, var, x_hat, gamma, eps)
    return out, cache
def d_layer_norm(dout, cache):
    x, mean, var, x_hat, gamma, eps = cache
    # dout 形状: (batch, ..., embed_dim)
    # 对除最后一个维度外的所有维度求和，得到 (embed_dim,)
    sum_axis = tuple(range(dout.ndim - 1))
    dgamma = np.sum(dout * x_hat, axis=sum_axis, keepdims=False)
    dbeta = np.sum(dout, axis=sum_axis, keepdims=False)
    # dx 计算保持不变，需要保持维度
    dx_hat = dout * gamma
    N = x.shape[-1]
    dx = (1.0 / N) * (1.0 / np.sqrt(var + eps)) * (
        N * dx_hat - np.sum(dx_hat, axis=-1, keepdims=True) - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
    )
    return dx, dgamma, dbeta
# ========== 本地实现 cross_entropy 和 d_cross_entropy（因为 funcs 中的版本未解包） ==========
def cross_entropy(probs, targets):
    """
    纯净的交叉熵损失（不包含 softmax）
    probs: (batch, seq, vocab) 概率分布（每行和为1）
    targets: (batch, seq) 整数索引
    返回：标量损失（平均）
    """
    batch, seq_len, vocab_size = probs.shape
    # 防止数值不稳定
    log_probs = np.log(probs + 1e-8)
    indices = (np.arange(batch)[:, None], np.arange(seq_len)[None, :], targets)
    loss = -np.mean(log_probs[indices])
    return loss

def d_cross_entropy(probs, targets):
    """
    返回损失对 probs 的梯度 (与 probs 同形状)
    """
    batch, seq_len, vocab_size = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch)[:, None], np.arange(seq_len)[None, :], targets] = 1.0
    # 梯度 = - (one_hot / probs) / (batch * seq_len)
    # 但为了数值稳定性，通常使用 - (one_hot / (probs + eps)) / N
    eps = 1e-8
    grad = - (one_hot / (probs + eps)) / (batch * seq_len)
    return grad



def matmul(a, b):
    return np.matmul(a, b)

def d_matmul(dout, a, b):
    da = np.matmul(dout, b.T)
    db = np.matmul(a.T, dout)
    return da, db
'''
def adam_update(params, m, v, grads, lr, step,beta1=0.9, beta2=0.999, eps=1e-8):
    for key in params:
        if key not in grads:
            continue
        m[key] = beta1 * m.get(key, 0) + (1 - beta1) * grads[key]
        v[key] = beta2 * v.get(key, 0) + (1 - beta2) * (grads[key] ** 2)
        params[key] -= lr * m[key] / (np.sqrt(v[key]) + eps)
    return params, m, v, step + 1
'''
def adam_update(params, grads, lr, step, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam 优化器
    params: 参数字典
    grads: 梯度字典（与 params 键对应）
    lr: 学习率
    step: 当前步数（从1开始）
    beta1: 一阶矩衰减率
    beta2: 二阶矩衰减率
    eps: 数值稳定小量
    返回: (更新后的 params, 新的 step)
    """
    # 初始化动量存储（如果尚未存在）
    if not hasattr(adam_update, 'm'):
        adam_update.m = {}
        adam_update.v = {}
    
    for key in params:
        # 如果 key 不在 grads 中，跳过（某些层可能没有梯度）
        if key not in grads:
            continue
        
        g = grads[key]
        
        # 如果该参数还没有动量，初始化
        if key not in adam_update.m:
            adam_update.m[key] = np.zeros_like(params[key])
            adam_update.v[key] = np.zeros_like(params[key])
        
        # 更新一阶矩和二阶矩
        adam_update.m[key] = beta1 * adam_update.m[key] + (1 - beta1) * g
        adam_update.v[key] = beta2 * adam_update.v[key] + (1 - beta2) * (g * g)
        
        # 偏差校正
        m_hat = adam_update.m[key] / (1 - beta1 ** step)
        v_hat = adam_update.v[key] / (1 - beta2 ** step)
        
        # 更新参数
        params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return params, step + 1
