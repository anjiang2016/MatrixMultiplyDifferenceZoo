import numpy as np
from numpy.lib.stride_tricks import as_strided

def conv(x, w, b=None, stride=1, padding=0):
    """
    卷积前向传播
    
    Args:
        x: (N, C, H, W) 输入
        w: (C_out, C, Hk, Wk) 权重
        b: (C_out,) 偏置，可选
        stride: int 或 tuple
        padding: int 或 tuple
    
    Returns:
        out: (N, C_out, H_out, W_out)
        x_pad: padding 后的输入（用于反向传播缓存）
        (H_out, W_out): 输出尺寸
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
    
    # Padding
    x_pad = np.pad(x, ((0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # 输出尺寸
    H_out = (H + 2*pad_h - Hk) // stride_h + 1
    W_out = (W + 2*pad_w - Wk) // stride_w + 1
    
    # im2col
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
    cols = cols.reshape(N, C, H_out * W_out, Hk * Wk)  # (N, C, M, Hk*Wk)
    
    # 矩阵乘法：使用 reshape + matmul
    cols = cols.transpose(0, 2, 1, 3).reshape(N * H_out * W_out, C * Hk * Wk)
    w_flat = w.reshape(C_out, C * Hk * Wk)  # (C_out, C*Hk*Wk)
    
    out = cols @ w_flat.T  # (N*M, C_out)
    out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)  # (N, C_out, H_out, W_out)
    
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    
    return out, x_pad, (H_out, W_out)


def d_conv(dout, x, w, stride=1, padding=0, x_pad=None, H_out=None, W_out=None):
    """
    卷积反向传播：计算 dx, dw, db
    
    Args:
        dout: (N, C_out, H_out, W_out) 上游梯度
        x: (N, C, H, W) 原始输入（未padding）
        w: (C_out, C, Hk, Wk) 权重
        stride: int 或 tuple
        padding: int 或 tuple
        x_pad: 前向传播缓存的 padding 后的输入
        H_out, W_out: 前向传播缓存的输出尺寸
    
    Returns:
        dx: (N, C, H, W) 输入梯度
        dw: (C_out, C, Hk, Wk) 权重梯度
        db: (C_out,) 偏置梯度（如果有）
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
    
    # 如果没有传入缓存，重新计算
    if x_pad is None or H_out is None or W_out is None:
        x_pad = np.pad(x, ((0,0), (0,0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        H_out = (H + 2*pad_h - Hk) // stride_h + 1
        W_out = (W + 2*pad_w - Wk) // stride_w + 1
    
    # ========== 1. 计算 db（偏置梯度）==========
    db = dout.sum(axis=(0, 2, 3))
    
    # ========== 2. 计算 dw（权重梯度）==========
    # im2col
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
    cols = cols.reshape(N, C, H_out * W_out, Hk * Wk)  # (N, C, M, Hk*Wk)
    cols = cols.transpose(0, 2, 1, 3).reshape(N * H_out * W_out, C * Hk * Wk)
    
    # dout 重塑
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(N * H_out * W_out, C_out)
    
    # dw = dout_flat.T @ cols
    dw = dout_flat.T @ cols  # (C_out, C*Hk*Wk)
    dw = dw.reshape(C_out, C, Hk, Wk)
    
    # ========== 3. 计算 dx（输入梯度）==========
    # w_flat: (C_out, C*Hk*Wk) -> (C*Hk*Wk, C_out)
    w_flat = w.reshape(C_out, C * Hk * Wk).T  # (C*Hk*Wk, C_out)
    
    # dcols = dout_flat @ w_flat.T
    dcols = dout_flat @ w_flat.T  # (N*M, C*Hk*Wk)
    
    # 重塑为 (N, H_out, W_out, C, Hk, Wk)
    dcols = dcols.reshape(N, H_out, W_out, C, Hk, Wk).transpose(0, 3, 1, 2, 4, 5)
    
    # col2im: 将梯度映射回输入空间
    dx_pad = np.zeros((N, C, H + 2*pad_h, W + 2*pad_w), dtype=dcols.dtype)
    
    # 累加梯度
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w_idx in range(W_out):
                    dx_pad[n, c, h*stride_h:h*stride_h+Hk, w_idx*stride_w:w_idx*stride_w+Wk] += dcols[n, c, h, w_idx]
    
    # 去掉 padding
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
