import numpy as np
import convmodule

def conv(x, w, b=None, stride=1, padding=0):
    """使用 C 扩展加速的卷积"""
    if b is not None:
        out = convmodule.conv_c(x, w, stride, padding)
        # 添加偏置（可以在 C 层做，但为了简化先保留）
        out += b.reshape(1, -1, 1, 1)
    else:
        out = convmodule.conv_c(x, w, stride, padding)
    
    # 缓存信息仍用 Python 方式（因为 C 层不缓存）
    x_pad = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    H_out = (x.shape[2] + 2*padding - w.shape[2]) // stride + 1
    W_out = (x.shape[3] + 2*padding - w.shape[3]) // stride + 1
    cache = (x, w, b, stride, padding, x_pad, H_out, W_out)
    return out, cache
