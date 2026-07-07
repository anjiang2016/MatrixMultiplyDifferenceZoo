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
