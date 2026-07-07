"""
fenge.py - 分割专用模块
依赖 funcs.py 中的基础层
"""

import numpy as np
from funcs import (
    conv, d_conv,
    relu, d_relu,
    sigmoid, d_sigmoid,
    avgpool, d_avgpool
)

def upsample_bilinear(x, scale_factor=2):
    """
    双线性插值上采样
    x: (N, C, H, W)
    scale_factor: float 放大倍数（可为小数）
    返回: (N, C, out_h, out_w), cache
    """
    N, C, H, W = x.shape
    out_h = int(round(H * scale_factor))
    out_w = int(round(W * scale_factor))
    
    if out_h == H and out_w == W:
        return x.copy(), (x.shape, out_h, out_w, scale_factor)
    
    h_scale = (H - 1) / (out_h - 1) if out_h > 1 else 0
    w_scale = (W - 1) / (out_w - 1) if out_w > 1 else 0
    
    h_coords = np.arange(out_h) * h_scale
    w_coords = np.arange(out_w) * w_scale
    
    h0 = np.floor(h_coords).astype(np.int32)
    h1 = np.minimum(h0 + 1, H - 1)
    w0 = np.floor(w_coords).astype(np.int32)
    w1 = np.minimum(w0 + 1, W - 1)
    
    h_frac = h_coords - h0
    w_frac = w_coords - w0
    
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    v00 = x[n, c, h0[i], w0[j]]
                    v01 = x[n, c, h0[i], w1[j]]
                    v10 = x[n, c, h1[i], w0[j]]
                    v11 = x[n, c, h1[i], w1[j]]
                    
                    top = v00 * (1 - w_frac[j]) + v01 * w_frac[j]
                    bottom = v10 * (1 - w_frac[j]) + v11 * w_frac[j]
                    out[n, c, i, j] = top * (1 - h_frac[i]) + bottom * h_frac[i]
    
    cache = (x.shape, out_h, out_w, scale_factor)
    return out, cache


def d_upsample_bilinear(dout, cache):
    """
    双线性插值上采样的反向传播
    """
    in_shape, out_h, out_w, scale = cache
    N, C, H, W = in_shape
    dx = np.zeros(in_shape, dtype=dout.dtype)
    
    h_scale = (H - 1) / (out_h - 1) if out_h > 1 else 0
    w_scale = (W - 1) / (out_w - 1) if out_w > 1 else 0
    
    h_coords = np.arange(out_h) * h_scale
    w_coords = np.arange(out_w) * w_scale
    
    h0 = np.floor(h_coords).astype(np.int32)
    h1 = np.minimum(h0 + 1, H - 1)
    w0 = np.floor(w_coords).astype(np.int32)
    w1 = np.minimum(w0 + 1, W - 1)
    
    h_frac = h_coords - h0
    w_frac = w_coords - w0
    
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    grad = dout[n, c, i, j]
                    dx[n, c, h0[i], w0[j]] += grad * (1 - h_frac[i]) * (1 - w_frac[j])
                    dx[n, c, h0[i], w1[j]] += grad * (1 - h_frac[i]) * w_frac[j]
                    dx[n, c, h1[i], w0[j]] += grad * h_frac[i] * (1 - w_frac[j])
                    dx[n, c, h1[i], w1[j]] += grad * h_frac[i] * w_frac[j]
    
    return dx
def dice_loss(pred, target, eps=1e-8):
    """
    Dice Loss，返回 (loss, cache)
    cache = (pred, target) 用于反向传播
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    
    intersection = (pred_flat * target_flat).sum(axis=1)
    union = pred_flat.sum(axis=1) + target_flat.sum(axis=1)
    dice = (2 * intersection + eps) / (union + eps)
    loss = 1 - dice.mean()
    
    cache = (pred, target)  # 缓存用于反向传播
    return loss, cache


def d_dice_loss(cache):
    """Dice Loss 的反向传播"""
    pred, target = cache
    eps = 1e-8
    
    N = pred.shape[0]
    pred_flat = pred.reshape(N, -1)
    target_flat = target.reshape(N, -1)
    
    A = (pred_flat * target_flat).sum(axis=1, keepdims=True)
    B = pred_flat.sum(axis=1, keepdims=True)
    C = target_flat.sum(axis=1, keepdims=True)
    denominator = B + C + eps
    
    grad_flat = (2 * A - 2 * target_flat * denominator) / (denominator ** 2)
    grad = grad_flat.reshape(pred.shape)
    
    return grad
def dice_loss_b(pred, target, eps=1e-8):
    """Dice Loss"""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(axis=1)
    union = pred_flat.sum(axis=1) + target_flat.sum(axis=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()
def binary_cross_entropy_with_dice(pred, target, alpha=0.5):
    """交叉熵 + Dice Loss 组合"""
    ce_loss, _ = binary_cross_entropy(pred, target)
    dc_loss = dice_lossi_b(pred, target)
    return alpha * ce_loss + (1 - alpha) * dc_loss
def d_binary_cross_entropy_with_dice(cache):
    """
    组合损失的反向传播
    """
    pred, target, alpha, ce_cache = cache
    
    # 交叉熵的梯度
    d_ce = d_binary_cross_entropy(ce_cache)
    
    # Dice Loss 的梯度
    d_dice = d_dice_loss(pred, target)
    
    # 组合梯度
    grad = alpha * d_ce + (1 - alpha) * d_dice
    
    return grad
# ============================================================
# 分割网络
# ============================================================
def init_seg_weights():
    """初始化对称分割网络权重"""
    weights = {}
    
    def he_init(shape):
        if len(shape) == 4:
            fan_in = shape[1] * shape[2] * shape[3]
        else:
            fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape).astype(np.float32) * std
    
    # ========== 编码器 ==========
    # Conv1: 1 -> 6, 5x5, pad=0
    weights['conv1_w'] = he_init((6, 1, 5, 5))
    weights['conv1_b'] = np.zeros(6, dtype=np.float32)
    
    # Conv2: 6 -> 16, 5x5, pad=0
    weights['conv2_w'] = he_init((16, 6, 5, 5))
    weights['conv2_b'] = np.zeros(16, dtype=np.float32)
    
    # Conv3: 1x1 conv, 16 -> 16 (保持5x5空间信息)
    weights['conv3_w'] = he_init((16, 16, 1, 1))
    weights['conv3_b'] = np.zeros(16, dtype=np.float32)
    
    # ========== 解码器 ==========
    # Conv4: (16+16)=32 -> 6, 3x3, pad=1
    weights['conv4_w'] = he_init((6, 32, 3, 3))
    weights['conv4_b'] = np.zeros(6, dtype=np.float32)
    
    # Conv5: (6+6)=12 -> 6, 3x3, pad=1
    weights['conv5_w'] = he_init((6, 12, 3, 3))
    weights['conv5_b'] = np.zeros(6, dtype=np.float32)
    
    # Conv6: (6+6)=12 -> 1, 3x3, pad=1
    weights['conv6_w'] = he_init((1, 12, 3, 3))
    weights['conv6_b'] = np.zeros(1, dtype=np.float32)
    
    return weights

def forward_seg(x, weights, training=True, verbose=False):
    """
    对称分割网络前向传播
    输入: (N, 1, 32, 32) -> 输出: (N, 1, 32, 32)
    """
    caches = {}
    
    if verbose:
        print("\n" + "="*70)
        print(f"{'Layer':<18} {'Output Shape':<20} {'Notes':<30}")
        print("="*70)
        print(f"{'Input':<18} {str(x.shape):<20}")
    
    # ============================================================
    # 编码器 (Encoder)
    # ============================================================
    
    # Conv1: 32x32 -> 28x28
    out, x_pad, (H_out, W_out) = conv(x, weights['conv1_w'], weights['conv1_b'], stride=1, padding=0)
    caches['conv1'] = (x, weights['conv1_w'], weights['conv1_b'], 1, 0, x_pad, H_out, W_out)
    if verbose:
        print(f"{'Conv1':<18} {str(out.shape):<20} 5x5, pad=0, 1->6")
    
    out = relu(out)
    caches['relu1'] = out
    if verbose:
        print(f"{'ReLU1':<18} {str(out.shape):<20}")
    
    # 保存 Conv1 输出用于跳跃连接
    skip1 = out  # (N, 6, 28, 28)
    caches['skip1'] = skip1
    
    # Pool1: 28x28 -> 14x14
    out, cache = avgpool(out, kernel_size=2, stride=2, padding=0)
    caches['pool1'] = cache
    if verbose:
        print(f"{'Pool1':<18} {str(out.shape):<20} 2x2, stride=2")
    
    # 保存 Pool1 输出用于跳跃连接
    skip2 = out  # (N, 6, 14, 14)
    caches['skip2'] = skip2
    
    # Conv2: 14x14 -> 10x10
    conv2_input = out
    out, x_pad, (H_out, W_out) = conv(conv2_input, weights['conv2_w'], weights['conv2_b'], stride=1, padding=0)
    caches['conv2'] = (conv2_input, weights['conv2_w'], weights['conv2_b'], 1, 0, x_pad, H_out, W_out)
    if verbose:
        print(f"{'Conv2':<18} {str(out.shape):<20} 5x5, pad=0, 6->16")
    
    out = relu(out)
    caches['relu2'] = out
    if verbose:
        print(f"{'ReLU2':<18} {str(out.shape):<20}")
    
    # 保存 Conv2 输出用于跳跃连接
    skip3 = out  # (N, 16, 10, 10)
    caches['skip3'] = skip3
    
    # Pool2: 10x10 -> 5x5
    out, cache = avgpool(out, kernel_size=2, stride=2, padding=0)
    caches['pool2'] = cache
    if verbose:
        print(f"{'Pool2':<18} {str(out.shape):<20} 2x2, stride=2")
    
    # Conv3: 1x1 conv, 16 -> 16, 保持 5x5
    conv3_input = out
    out, x_pad, (H_out, W_out) = conv(conv3_input, weights['conv3_w'], weights['conv3_b'], stride=1, padding=0)
    caches['conv3'] = (conv3_input, weights['conv3_w'], weights['conv3_b'], 1, 0, x_pad, H_out, W_out)
    if verbose:
        print(f"{'Conv3':<18} {str(out.shape):<20} 1x1 conv, 16->16 (瓶颈)")
    
    out = relu(out)
    caches['relu3'] = out
    if verbose:
        print(f"{'ReLU3':<18} {str(out.shape):<20} ← 瓶颈 (5x5x16)")
    
    # ============================================================
    # 解码器 (Decoder) - 与编码器对称
    # ============================================================
    
    # Upsample1: 5x5 -> 10x10
    out, cache = upsample_bilinear(out, scale_factor=2)
    caches['upsample1'] = cache
    if verbose:
        print(f"{'Upsample1':<18} {str(out.shape):<20} scale=2, 5->10")
    
    # 跳跃连接 3: 拼接 Conv2 输出 (N, 16, 10, 10)
    out = np.concatenate([out, skip3], axis=1)  # (N, 16+16=32, 10, 10)
    caches['concat1'] = out
    if verbose:
        print(f"{'Concat1':<18} {str(out.shape):<20} + skip3 (16+16=32)")
    
    # Conv4: 32 -> 6, 3x3, pad=1
    conv4_input = out
    out, x_pad, (H_out, W_out) = conv(conv4_input, weights['conv4_w'], weights['conv4_b'], stride=1, padding=1)
    caches['conv4'] = (conv4_input, weights['conv4_w'], weights['conv4_b'], 1, 1, x_pad, H_out, W_out)
    if verbose:
        print(f"{'Conv4':<18} {str(out.shape):<20} 3x3, pad=1, 32->6")
    
    out = relu(out)
    caches['relu4'] = out
    if verbose:
        print(f"{'ReLU4':<18} {str(out.shape):<20}")
    
    # Upsample2: 10x10 -> 14x14 (scale=1.4)
    out, cache = upsample_bilinear(out, scale_factor=1.4)
    caches['upsample2'] = cache
    if verbose:
        print(f"{'Upsample2':<18} {str(out.shape):<20} scale=1.4, 10->14")
    
    # 跳跃连接 2: 拼接 Pool1 输出 (N, 6, 14, 14)
    out = np.concatenate([out, skip2], axis=1)  # (N, 6+6=12, 14, 14)
    caches['concat2'] = out
    if verbose:
        print(f"{'Concat2':<18} {str(out.shape):<20} + skip2 (6+6=12)")
    
    # Conv5: 12 -> 6, 3x3, pad=1
    conv5_input = out
    out, x_pad, (H_out, W_out) = conv(conv5_input, weights['conv5_w'], weights['conv5_b'], stride=1, padding=1)
    caches['conv5'] = (conv5_input, weights['conv5_w'], weights['conv5_b'], 1, 1, x_pad, H_out, W_out)
    if verbose:
        print(f"{'Conv5':<18} {str(out.shape):<20} 3x3, pad=1, 12->6")
    
    out = relu(out)
    caches['relu5'] = out
    if verbose:
        print(f"{'ReLU5':<18} {str(out.shape):<20}")
    
    # Upsample3: 14x14 -> 28x28 (scale=2)
    out, cache = upsample_bilinear(out, scale_factor=2)
    caches['upsample3'] = cache
    if verbose:
        print(f"{'Upsample3':<18} {str(out.shape):<20} scale=2, 14->28")
    
    # 跳跃连接 1: 拼接 Conv1 输出 (N, 6, 28, 28)
    out = np.concatenate([out, skip1], axis=1)  # (N, 6+6=12, 28, 28)
    caches['concat3'] = out
    if verbose:
        print(f"{'Concat3':<18} {str(out.shape):<20} + skip1 (6+6=12)")
    
    # Conv6: 12 -> 1, 3x3, pad=1
    conv6_input = out
    out, x_pad, (H_out, W_out) = conv(conv6_input, weights['conv6_w'], weights['conv6_b'], stride=1, padding=1)
    caches['conv6'] = (conv6_input, weights['conv6_w'], weights['conv6_b'], 1, 1, x_pad, H_out, W_out)
    if verbose:
        print(f"{'Conv6':<18} {str(out.shape):<20} 3x3, pad=1, 12->1")
    
    out = relu(out)
    caches['relu6'] = out
    if verbose:
        print(f"{'ReLU6':<18} {str(out.shape):<20}")
    
    # Upsample4: 28x28 -> 32x32 (scale=1.142857)
    out, cache = upsample_bilinear(out, scale_factor=32/28)  # ≈1.142857
    caches['upsample4'] = cache
    if verbose:
        print(f"{'Upsample4':<18} {str(out.shape):<20} scale=32/28, 28->32")
    
    # Sigmoid 输出
    out = sigmoid(out)
    caches['sigmoid_out'] = out
    if verbose:
        print(f"{'Sigmoid':<18} {str(out.shape):<20} 最终输出")
        print("="*70)
    
    return out, caches

def backward_seg(dout, caches, weights):
    """
    对称分割网络反向传播
    严格按照前向传播的逆序，正确累加跳跃连接梯度
    """
    grads = {}
    
    # ============================================================
    # 1. 解码器反向 (Decoder Backward)
    # ============================================================
    
    # Sigmoid 反向
    dx = dout * d_sigmoid(None, caches['sigmoid_out'])
    
    # Upsample4 反向: 32x32 -> 28x28
    dx = d_upsample_bilinear(dx, caches['upsample4'])
    
    # ReLU6 反向
    dx = dx * d_relu(None, caches['relu6'])
    
    # Conv6 反向: 输入 concat3 (12, 28, 28)
    x, w, b, stride, padding, x_pad, H_out, W_out = caches['conv6']
    dx, dw, db = d_conv(dx, x, w, stride, padding, x_pad, H_out, W_out)
    grads['conv6_w'] = dw
    grads['conv6_b'] = db
    # dx shape: (N, 12, 28, 28)
    
    # 拆分 concat3 的梯度
    dx_upsample3 = dx[:, :6, :, :]   # (N, 6, 28, 28) 给 Upsample3
    dx_skip1 = dx[:, 6:, :, :]       # (N, 6, 28, 28) 给 skip1 (Conv1 输出)
    
    # Upsample3 反向: 28x28 -> 14x14
    dx = d_upsample_bilinear(dx_upsample3, caches['upsample3'])
    
    # ReLU5 反向
    dx = dx * d_relu(None, caches['relu5'])
    
    # Conv5 反向: 输入 concat2 (12, 14, 14)
    x, w, b, stride, padding, x_pad, H_out, W_out = caches['conv5']
    dx, dw, db = d_conv(dx, x, w, stride, padding, x_pad, H_out, W_out)
    grads['conv5_w'] = dw
    grads['conv5_b'] = db
    # dx shape: (N, 12, 14, 14)
    
    # 拆分 concat2 的梯度
    dx_upsample2 = dx[:, :6, :, :]   # (N, 6, 14, 14) 给 Upsample2
    dx_skip2 = dx[:, 6:, :, :]       # (N, 6, 14, 14) 给 skip2 (Pool1 输出)
    
    # Upsample2 反向: 14x14 -> 10x10
    dx = d_upsample_bilinear(dx_upsample2, caches['upsample2'])
    
    # ReLU4 反向
    dx = dx * d_relu(None, caches['relu4'])
    
    # Conv4 反向: 输入 concat1 (32, 10, 10)
    x, w, b, stride, padding, x_pad, H_out, W_out = caches['conv4']
    dx, dw, db = d_conv(dx, x, w, stride, padding, x_pad, H_out, W_out)
    grads['conv4_w'] = dw
    grads['conv4_b'] = db
    # dx shape: (N, 32, 10, 10)
    
    # 拆分 concat1 的梯度
    dx_upsample1 = dx[:, :16, :, :]  # (N, 16, 10, 10) 给 Upsample1
    dx_skip3 = dx[:, 16:, :, :]      # (N, 16, 10, 10) 给 skip3 (Conv2 输出)
    
    # Upsample1 反向: 10x10 -> 5x5
    dx = d_upsample_bilinear(dx_upsample1, caches['upsample1'])
    
    # ReLU3 反向 (瓶颈)
    dx = dx * d_relu(None, caches['relu3'])
    
    # ============================================================
    # 2. 编码器反向 (Encoder Backward)
    # ============================================================
    
    # Conv3 反向: 输入瓶颈梯度 dx (16, 5, 5)
    x, w, b, stride, padding, x_pad, H_out, W_out = caches['conv3']
    dx, dw, db = d_conv(dx, x, w, stride, padding, x_pad, H_out, W_out)
    grads['conv3_w'] = dw
    grads['conv3_b'] = db
    # dx shape: (N, 16, 5, 5) 这是 Pool2 输出的梯度
    
    # Pool2 反向: 5x5 -> 10x10
    dx = d_avgpool(dx, caches['pool2'])
    # dx shape: (N, 16, 10, 10) 这是 Conv2 输出的梯度
    
    # 合并 skip3 的梯度 (来自跳跃连接)
    dx += dx_skip3  # 现在 dx 是 Conv2 输出的完整梯度
    
    # ReLU2 反向 (Conv2 输出经过 ReLU2)
    dx = dx * d_relu(None, caches['relu2'])
    
    # Conv2 反向: 输入是合并后的梯度 (16, 10, 10)
    x, w, b, stride, padding, x_pad, H_out, W_out = caches['conv2']
    dx, dw, db = d_conv(dx, x, w, stride, padding, x_pad, H_out, W_out)
    grads['conv2_w'] = dw
    grads['conv2_b'] = db
    # dx shape: (N, 6, 14, 14) 这是 Pool1 输出的梯度
    
    # 合并 skip2 的梯度 (来自跳跃连接)
    dx += dx_skip2  # 现在 dx 是 Pool1 输出的完整梯度
    
    # Pool1 反向: 14x14 -> 28x28
    dx = d_avgpool(dx, caches['pool1'])
    # dx shape: (N, 6, 28, 28) 这是 Conv1 输出的梯度
    
    # 合并 skip1 的梯度 (来自跳跃连接)
    dx += dx_skip1  # 现在 dx 是 Conv1 输出的完整梯度
    
    # ReLU1 反向 (Conv1 输出经过 ReLU1)
    dx = dx * d_relu(None, caches['relu1'])
    
    # Conv1 反向: 输入是合并后的梯度 (6, 28, 28)
    x, w, b, stride, padding, x_pad, H_out, W_out = caches['conv1']
    dx, dw, db = d_conv(dx, x, w, stride, padding, x_pad, H_out, W_out)
    grads['conv1_w'] = dw
    grads['conv1_b'] = db
    # dx shape: (N, 1, 32, 32) 最终输入梯度
    
    return grads

# ============================================================
# 分割损失函数
# ============================================================
'''
def binary_cross_entropy(pred, target):
    """
    逐像素二分类交叉熵
    pred: (N, 1, H, W) 预测概率 (经过 sigmoid)
    target: (N, 1, H, W) 真实标签 (0 或 1)
    """
    eps = 1e-12
    pred_clipped = np.clip(pred, eps, 1 - eps)
    loss = -np.mean(target * np.log(pred_clipped) + (1 - target) * np.log(1 - pred_clipped))
    cache = (pred, target)
    return loss, cache


def d_binary_cross_entropy(cache):
    """二分类交叉熵反向传播"""
    pred, target = cache
    eps = 1e-12
    pred_clipped = np.clip(pred, eps, 1 - eps)
    return (pred_clipped - target) / (pred_clipped * (1 - pred_clipped)) / pred.size
'''
def binary_cross_entropy(pred, target, pos_weight=6.0):
    """
    加权二分类交叉熵
    pos_weight: 正样本权重（前景），越大越重视前景
    """
    eps = 1e-12
    pred_clipped = np.clip(pred, eps, 1 - eps)

    # 加权损失：前景权重提高
    loss = -np.mean(
        pos_weight * target * np.log(pred_clipped) +
        (1 - target) * np.log(1 - pred_clipped)
    )
    cache = (pred, target, pos_weight)
    return loss, cache


def d_binary_cross_entropy(cache):
    pred, target, pos_weight = cache
    eps = 1e-12
    pred_clipped = np.clip(pred, eps, 1 - eps)
    # 加权梯度
    grad = (pos_weight * target * (pred_clipped - 1) + (1 - target) * pred_clipped) / (pred_clipped * (1 - pred_clipped))
    return grad / pred.size

def generate_seg_labels(X, threshold=0.5):
    """从 MNIST 图像生成二值分割标签（前景=1，背景=0）"""
    return (X > threshold).astype(np.float32)

def compute_iou(pred, target, threshold=0.5):
    """计算 IoU（交并比）"""
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)
    intersection = (pred_binary * target_binary).sum()
    union = (pred_binary + target_binary).sum() - intersection + 1e-8
    return intersection / union
import pickle
import os
import time

def save_seg_model(weights, filepath, epoch=None, loss=None, iou=None, optimizer_state=None):
    """
    保存分割模型权重和训练状态
    
    Args:
        weights: 模型权重字典
        filepath: 保存路径（如 'models/seg_model.npz'）
        epoch: 当前 epoch（可选）
        loss: 当前 loss（可选）
        iou: 当前 IoU（可选）
        optimizer_state: 优化器状态（可选，用于 Adam 等）
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # 准备保存的数据
    save_dict = {}
    
    # 保存权重
    for key, value in weights.items():
        save_dict[f'weight_{key}'] = value
    
    # 保存训练状态（如果有）
    if epoch is not None:
        save_dict['epoch'] = epoch
    if loss is not None:
        save_dict['loss'] = loss
    if iou is not None:
        save_dict['iou'] = iou
    if optimizer_state is not None:
        save_dict['optimizer_state'] = optimizer_state
    
    # 保存时间戳
    save_dict['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save_dict['version'] = '1.0'
    
    np.savez(filepath, **save_dict)
    print(f"✅ 模型已保存到: {filepath}")
    if epoch is not None:
        print(f"   Epoch: {epoch}, Loss: {loss:.4f}, IoU: {iou:.4f}")


def load_seg_model(filepath):
    """
    加载分割模型权重和训练状态
    
    Args:
        filepath: 模型文件路径
    
    Returns:
        weights: 权重字典
        metadata: 训练状态字典（epoch, loss, iou, timestamp 等）
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    
    data = np.load(filepath, allow_pickle=True)
    
    # 提取权重
    weights = {}
    for key in data.files:
        if key.startswith('weight_'):
            weight_name = key[7:]  # 去掉 'weight_' 前缀
            weights[weight_name] = data[key]
    
    # 提取元数据
    metadata = {}
    for key in ['epoch', 'loss', 'iou', 'timestamp', 'version', 'optimizer_state']:
        if key in data.files:
            metadata[key] = data[key].item() if data[key].size == 1 else data[key]
    
    print(f"✅ 模型已加载: {filepath}")
    if 'epoch' in metadata:
        print(f"   Epoch: {metadata['epoch']}, Loss: {metadata.get('loss', 'N/A')}, IoU: {metadata.get('iou', 'N/A')}")
    if 'timestamp' in metadata:
        print(f"   保存时间: {metadata['timestamp']}")
    
    return weights, metadata


def list_saved_models(directory='models'):
    """列出所有保存的模型"""
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return []
    
    files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    if not files:
        print(f"没有找到模型文件")
        return []
    
    print(f"\n找到 {len(files)} 个模型文件:")
    for f in sorted(files, reverse=True):
        filepath = os.path.join(directory, f)
        try:
            data = np.load(filepath, allow_pickle=True)
            epoch = data.get('epoch', '?')
            iou = data.get('iou', '?')
            timestamp = data.get('timestamp', '?')
            print(f"  {f} - Epoch: {epoch}, IoU: {iou}, Time: {timestamp}")
        except:
            print(f"  {f} - (读取失败)")
    
    return files
