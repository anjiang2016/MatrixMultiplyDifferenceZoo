"""
train_diffusion_single_full.py
单张图片 + 时间嵌入加到所有层级（通道匹配 + 线性层形状修正）
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funcs import conv, d_conv, relu, d_relu, linear, d_linear
import os
import numpy as np
from PIL import Image
import glob
from PIL import Image   # 仅用于读取图片，可换为 cv2

def save_checkpoint(filepath, weights, m=None, v=None, epoch=None, losses=None, optimizer='adam'):
    """保存检查点，m,v 可选（若为None则不存）"""
    data = {
        'weights': weights,
        'epoch': epoch,
        'losses': losses,
        'optimizer': optimizer
    }
    if m is not None:
        data['m'] = m
    if v is not None:
        data['v'] = v
    np.savez(filepath, **data)

def load_checkpoint(filepath):
    """加载检查点，返回 (weights, m, v, epoch, losses)，若文件不存在返回 None"""
    if not os.path.exists(filepath):
        return None, None, None, 0, []
    data = np.load(filepath, allow_pickle=True)
    weights = data['weights'].item()
    m = data['m'].item() if 'm' in data else None
    v = data['v'].item() if 'v' in data else None
    epoch = int(data['epoch'])
    losses = list(data['losses'])
    return weights, m, v, epoch, losses

def cosine_annealing_lr(epoch, total_epochs, lr_max=1e-3, lr_min=1e-6):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / total_epochs * np.pi))
# ============================================================
# 1. 时间嵌入 & 噪声调度
# ============================================================
def get_timestep_embedding(t, emb_dim=64):
    half_dim = emb_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    return emb


def linear_beta_schedule(timesteps=100):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)


def get_sqrt_alphas_bar(betas):
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    return np.sqrt(alphas_bar), np.sqrt(1.0 - alphas_bar)


def upsample_nearest(x, scale_factor=2):
    N, C, H, W = x.shape
    return x.repeat(scale_factor, axis=2).repeat(scale_factor, axis=3)
def batch_norm(x, gamma, beta, eps=1e-5):
    """
    x: (N, C, H, W)
    gamma: (C,)
    beta: (C,)
    返回: out, cache (包含用于反向的变量)
    """
    N, C, H, W = x.shape
    # 计算每个通道的均值和方差
    mean = x.mean(axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1)
    var = x.var(axis=(0, 2, 3), keepdims=True)     # (1, C, 1, 1)
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma.reshape(1, C, 1, 1) * x_hat + beta.reshape(1, C, 1, 1)
    cache = (x, mean, var, x_hat, gamma, eps)
    return out, cache
def d_batch_norm(dout, cache):
    x, mean, var, x_hat, gamma, eps = cache
    N, C, H, W = x.shape
    # 计算梯度
    dgamma = (dout * x_hat).sum(axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1)
    dbeta = dout.sum(axis=(0, 2, 3), keepdims=True)              # (1, C, 1, 1)
    dx_hat = dout * gamma.reshape(1, C, 1, 1)
    dx = (1.0 / (N * H * W)) * (1.0 / np.sqrt(var + eps)) * (
        N * H * W * dx_hat - dx_hat.sum(axis=(0,2,3), keepdims=True) - x_hat * (dx_hat * x_hat).sum(axis=(0,2,3), keepdims=True)
    )
    return dx, dgamma.reshape(C), dbeta.reshape(C)
# ============================================================
# 2. 模型定义（时间嵌入加到每层，通道匹配）
# ============================================================
def init_diffusion_weights():
    weights = {}

    def he_init(shape):
        fan_in = np.prod(shape[1:]) if len(shape) > 1 else shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape).astype(np.float32) * std

    # 编码器
    weights['enc1_w'] = he_init((16, 1, 3, 3))     # 输入1，输出16
    weights['enc1_b'] = np.zeros(16, dtype=np.float32)
    weights['enc2_w'] = he_init((32, 32, 3, 3))    # 输入16+16=32，输出32
    weights['enc2_b'] = np.zeros(32, dtype=np.float32)
    weights['enc3_w'] = he_init((64, 64, 3, 3))    # 输入32+32=64，输出64
    weights['enc3_b'] = np.zeros(64, dtype=np.float32)

    # 瓶颈
    weights['bn_w'] = he_init((64, 128, 3, 3))     # 输入64+64=128，输出64
    weights['bn_b'] = np.zeros(64, dtype=np.float32)

    # 解码器
    weights['dec3_w'] = he_init((64, 192, 3, 3))   # up(128) + skip2(64) = 192，输出64
    weights['dec3_b'] = np.zeros(64, dtype=np.float32)
    weights['dec2_w'] = he_init((32, 160, 3, 3))    # up(128) + skip1(32) = 160，输出32
    weights['dec2_b'] = np.zeros(32, dtype=np.float32)
    weights['dec1_w'] = he_init((1, 65, 3, 3))     # out(32) + x(1) = 33，输出1
    weights['dec1_b'] = np.zeros(1, dtype=np.float32)

    # 时间 FC 层（输入64，输出对应层通道数）
    weights['time_fc_enc1_w'] = he_init((64, 16))
    weights['time_fc_enc1_b'] = np.zeros(16, dtype=np.float32)
    weights['time_fc_enc2_w'] = he_init((64, 32))
    weights['time_fc_enc2_b'] = np.zeros(32, dtype=np.float32)
    weights['time_fc_enc3_w'] = he_init((64, 64))
    weights['time_fc_enc3_b'] = np.zeros(64, dtype=np.float32)
    weights['time_fc_bn_w'] = he_init((64, 64))
    weights['time_fc_bn_b'] = np.zeros(64, dtype=np.float32)
    weights['time_fc_dec3_w'] = he_init((64, 64))
    weights['time_fc_dec3_b'] = np.zeros(64, dtype=np.float32)
    weights['time_fc_dec2_w'] = he_init((64, 32))
    weights['time_fc_dec2_b'] = np.zeros(32, dtype=np.float32)

	# 为每个卷积层添加 BN 参数
    weights['bn_bn_gamma'] = np.ones(64, dtype=np.float32)
    weights['bn_bn_beta'] = np.zeros(64, dtype=np.float32)
    weights['bn_enc3_gamma'] = np.ones(64, dtype=np.float32)
    weights['bn_enc3_beta'] = np.zeros(64, dtype=np.float32)
    weights['bn_enc2_gamma'] = np.ones(32, dtype=np.float32)
    weights['bn_enc2_beta'] = np.zeros(32, dtype=np.float32)
    weights['bn_enc1_gamma'] = np.ones(16, dtype=np.float32)
    weights['bn_enc1_beta'] = np.zeros(16, dtype=np.float32)
    weights['bn_dec3_gamma'] = np.ones(64, dtype=np.float32)
    weights['bn_dec3_beta'] = np.zeros(64, dtype=np.float32)
    weights['bn_dec2_gamma'] = np.ones(32, dtype=np.float32)
    weights['bn_dec2_beta'] = np.zeros(32, dtype=np.float32)
    weights['bn_dec1_gamma'] = np.ones(1, dtype=np.float32)
    weights['bn_dec1_beta'] = np.zeros(1, dtype=np.float32)
    # 类似为 enc2, enc3, bn, dec3, dec2, dec1 添加（注意dec1输出通道为1）

    return weights
def forward_diffusion(x, t, weights):
    caches = {}
    time_emb = get_timestep_embedding(t, 64)
    caches['time_emb'] = time_emb

    def apply_time_feat(name, feat_map):
        w_key = f'time_fc_{name}_w'
        b_key = f'time_fc_{name}_b'
        time_feat_raw, _ = linear(time_emb, weights[w_key], weights[b_key])   # (N, out_dim)
        time_feat = relu(time_feat_raw)                                       # (N, out_dim)
        caches[f'time_feat_raw_{name}'] = time_feat_raw                      # 新增：存储 ReLU 前输出，用于掩码
        caches[f'time_feat_{name}'] = time_feat                              # 存储 ReLU 后（未广播）以备后续
        time_feat_exp = time_feat[:, :, None, None]                          # (N, out_dim, 1, 1)
        time_feat_exp = np.broadcast_to(time_feat_exp, (time_feat_exp.shape[0], time_feat_exp.shape[1], feat_map.shape[2], feat_map.shape[3]))
        caches[f'time_feat_broadcast_{name}'] = time_feat_exp               # 广播后的时间特征，用于拆分时定位
        return np.concatenate([feat_map, time_feat_exp], axis=1)             # 特征在前，时间在后

    # Enc1
    out, x_pad, (H_out, W_out) = conv(x, weights['enc1_w'], weights['enc1_b'], stride=1, padding=1)
    caches['enc1'] = (x, weights['enc1_w'], weights['enc1_b'], 1, 1, x_pad, H_out, W_out)
	# ---- 插入 BN ----
    out, bn_cache = batch_norm(out, weights['bn_enc1_gamma'], weights['bn_enc1_beta'])
    caches['bn_enc1'] = bn_cache
    out = relu(out)
    caches['relu_enc1_before_time'] = out
    out = apply_time_feat('enc1', out)  # 16+16=32
    skip1 = out
    caches['skip1'] = skip1
    caches['relu_enc1'] = out

    # Enc2
    conv2_input = out
    out, x_pad, (H_out, W_out) = conv(conv2_input, weights['enc2_w'], weights['enc2_b'], stride=2, padding=1)
    caches['enc2'] = (conv2_input, weights['enc2_w'], weights['enc2_b'], 2, 1, x_pad, H_out, W_out)
	# ---- 插入 BN ----
    out, bn_cache = batch_norm(out, weights['bn_enc2_gamma'], weights['bn_enc2_beta'])
    caches['bn_enc2'] = bn_cache
    out = relu(out)
    caches['relu_enc2_before_time'] = out
    out = apply_time_feat('enc2', out)  # 32+32=64
    skip2 = out
    caches['skip2'] = skip2
    caches['relu_enc2'] = out

    # Enc3
    conv3_input = out
    out, x_pad, (H_out, W_out) = conv(conv3_input, weights['enc3_w'], weights['enc3_b'], stride=2, padding=1)
    caches['enc3'] = (conv3_input, weights['enc3_w'], weights['enc3_b'], 2, 1, x_pad, H_out, W_out)
	# ---- 插入 BN ----
    out, bn_cache = batch_norm(out, weights['bn_enc3_gamma'], weights['bn_enc3_beta'])
    caches['bn_enc3'] = bn_cache
    out = relu(out)
    caches['relu_enc3_before_time'] = out
    out = apply_time_feat('enc3', out)  # 64+64=128
    caches['relu_enc3'] = out

    # 瓶颈
    bn_input = out
    out, x_pad, (H_out, W_out) = conv(bn_input, weights['bn_w'], weights['bn_b'], stride=1, padding=1)
    caches['bn'] = (bn_input, weights['bn_w'], weights['bn_b'], 1, 1, x_pad, H_out, W_out)
	# ---- 插入 BN ----
    out, bn_cache = batch_norm(out, weights['bn_bn_gamma'], weights['bn_bn_beta'])
    caches['bn_bn'] = bn_cache
    out = relu(out)
    caches['relu_bn_before_time'] = out
    out = apply_time_feat('bn', out)  # 64+64=128
    caches['relu_bn'] = out

    # Dec3
    up3 = upsample_nearest(out, 2)  # out: (N,128,8,8) -> (N,128,16,16)?? wait, out 是128通道？不对！
    # 重新检查：enc3 输出是64，拼接时间64后为128，但 apply_time_feat 返回拼接后的特征，所以 out 是128通道。
    # 但 up3 上采样后通道不变，仍然是128。
    # 但 skip2 是 enc2 拼接时间后的结果，64通道。
    # concat3 = 128 + 64 = 192？但我的权重初始化 dec3_w 是 (32, 192, 3,3)，匹配。
    # 所以 dec3_input 通道为192。
    concat3 = np.concatenate([up3, skip2], axis=1)
    caches['concat3'] = concat3
    dec3_input = concat3
    out, x_pad, (H_out, W_out) = conv(dec3_input, weights['dec3_w'], weights['dec3_b'], stride=1, padding=1)
    caches['dec3'] = (dec3_input, weights['dec3_w'], weights['dec3_b'], 1, 1, x_pad, H_out, W_out)
	# ---- 插入 BN ----
    out, bn_cache = batch_norm(out, weights['bn_dec3_gamma'], weights['bn_dec3_beta'])
    caches['bn_dec3'] = bn_cache
    out = relu(out)
    caches['relu_dec3_before_time'] = out  # 32通道
    out = apply_time_feat('dec3', out)  # 32+32=64
    caches['relu_dec3'] = out

    # Dec2
    up2 = upsample_nearest(out, 2)  # out: (N,64,16,16) -> (N,64,32,32)
    skip1  # (N,32,32,32)
    concat2 = np.concatenate([up2, skip1], axis=1)  # 64+32=96
    caches['concat2'] = concat2
    dec2_input = concat2
    out, x_pad, (H_out, W_out) = conv(dec2_input, weights['dec2_w'], weights['dec2_b'], stride=1, padding=1)
    caches['dec2'] = (dec2_input, weights['dec2_w'], weights['dec2_b'], 1, 1, x_pad, H_out, W_out)
	# ---- 插入 BN ----
    out, bn_cache = batch_norm(out, weights['bn_dec2_gamma'], weights['bn_dec2_beta'])
    caches['bn_dec2'] = bn_cache
    out = relu(out)
    caches['relu_dec2_before_time'] = out  # 16通道
    out = apply_time_feat('dec2', out)  # 16+16=32
    caches['relu_dec2'] = out

    # Dec1
    concat1 = np.concatenate([out, x], axis=1)  # 32 + 1 = 33
    caches['concat1'] = concat1
    dec1_input = concat1
    out, x_pad, (H_out, W_out) = conv(dec1_input, weights['dec1_w'], weights['dec1_b'], stride=1, padding=1)
    caches['dec1'] = (dec1_input, weights['dec1_w'], weights['dec1_b'], 1, 1, x_pad, H_out, W_out)
	# ---- 插入 BN ----
#    out, bn_cache = batch_norm(out, weights['bn_dec1_gamma'], weights['bn_dec1_beta'])
#    caches['bn_dec1'] = bn_cache
#    out = relu(out)
#    caches['relu_dec1'] = out  # 1通道，无时间

    return out, caches

def backward_diffusion(dout, caches, weights):
    grads = {}
    dx = dout   # (N, 1, 32, 32)
    
    # drelu
#    dx = d_relu(None,caches['relu_dec1'])*dx
    # 例如 Enc1 的反向（注意顺序：dout 是来自上层的梯度，经过 ReLU 反向后得到 dx_before_relu，然后经过 BN 反向得到 dx_before_bn，再传给 d_conv）
    # 在原来 `dx = dx_enc1_feat * d_relu(...)` 之后，添加：
#    dx_before_bn, dgamma, dbeta = d_batch_norm(dx, caches['bn_dec1'])
#    grads['bn_dec1_gamma'] = dgamma
#    grads['bn_dec1_beta'] = dbeta
#    dx = dx_before_bn   # 然后传给 d_conv
    # ---------- Dec1 (无时间) ----------
    # 输入 concat1 = [dec2_out (32) , x (1)] -> 33 通道
    dec1_input, w, b, stride, padding, x_pad, H_out, W_out = caches['dec1']
    dx_concat1, dw_dec1, db_dec1 = d_conv(dx, dec1_input, w, stride, padding, x_pad, H_out, W_out)
    grads['dec1_w'] = dw_dec1
    grads['dec1_b'] = db_dec1
    # 拆分：前32是 dec2 输出，后1是 x（忽略）
    dx_dec2_out = dx_concat1[:, :64, :, :]   # (N, 32, 32, 32)

    # dec2_out 结构：特征16 + 时间16
    dx_dec2_feat = dx_dec2_out[:, :32, :, :]          # 特征部分
    dx_time_dec2 = dx_dec2_out[:, 32:, :, :]          # 时间部分（空间梯度）
    caches['dx_time_dec2'] = dx_time_dec2.sum(axis=(2,3))  # 空间求和 -> (N, 16)

    # 特征部分通过 ReLU 导数（relu_dec2_before_time 为卷积输出 relu 后，即未拼接时间的激活）
    dx = dx_dec2_feat * d_relu(None, caches['relu_dec2_before_time'])

    dx_before_bn, dgamma, dbeta = d_batch_norm(dx, caches['bn_dec2'])
    grads['bn_dec2_gamma'] = dgamma
    grads['bn_dec2_beta'] = dbeta
    dx = dx_before_bn

    # ---------- Dec2 ----------
    # 输入 concat2 = [up2 (64), skip1 (32)] -> 96 通道
    dec2_input, w, b, stride, padding, x_pad, H_out, W_out = caches['dec2']
    dx_concat2, dw_dec2, db_dec2 = d_conv(dx, dec2_input, w, stride, padding, x_pad, H_out, W_out)
    grads['dec2_w'] = dw_dec2
    grads['dec2_b'] = db_dec2

    # 拆分 concat2 梯度
    dx_up2 = dx_concat2[:, :128, :, :]                 # up2 梯度 (N, 64, 32, 32)
    dx_skip1 = dx_concat2[:, 128:, :, :]               # skip1 梯度 (N, 32, 32, 32)

    # --- skip1 拆分 ---
    dx_skip1_feat = dx_skip1[:, :16, :, :]            # 特征部分 (N, 16, 32, 32)
    dx_time_enc1_from_skip = dx_skip1[:, 16:, :, :].sum(axis=(2,3))  # 空间求和 -> (N, 16)

    # --- up2 反向（最近邻上采样 2x）---
    N, C, H, W = dx_up2.shape   # (N,64,32,32)
    # 将梯度累加到原始 16x16 网格的对应位置
    dx_upsampled = np.zeros((N, C, H//2, W//2), dtype=dx_up2.dtype)
    for i in range(2):
        for j in range(2):
            dx_upsampled += dx_up2[:, :, i::2, j::2]
    # 不除以 4，保持梯度量级一致（若前向为复制，反向为求和）
    dx_dec3_out = dx_upsampled   # (N, 64, 16, 16)

    # dec3_out 结构：特征32 + 时间32
    dx_dec3_feat = dx_dec3_out[:, :64, :, :]          # 特征部分
    dx_time_dec3 = dx_dec3_out[:, 64:, :, :].sum(axis=(2,3))  # 空间求和 -> (N, 32)
    caches['dx_time_dec3'] = dx_time_dec3

    # 特征部分通过 ReLU 导数
    dx = dx_dec3_feat * d_relu(None, caches['relu_dec3_before_time'])
    dx_before_bn, dgamma, dbeta = d_batch_norm(dx, caches['bn_dec3'])
    grads['bn_dec3_gamma'] = dgamma
    grads['bn_dec3_beta'] = dbeta
    dx=dx_before_bn
    # ---------- Dec3 ----------
    # 输入 concat3 = [up3 (128), skip2 (64)] -> 192 通道
    dec3_input, w, b, stride, padding, x_pad, H_out, W_out = caches['dec3']
    dx_concat3, dw_dec3, db_dec3 = d_conv(dx, dec3_input, w, stride, padding, x_pad, H_out, W_out)
    grads['dec3_w'] = dw_dec3
    grads['dec3_b'] = db_dec3

    dx_up3 = dx_concat3[:, :128, :, :]                # up3 梯度 (N, 128, 16, 16)
    dx_skip2 = dx_concat3[:, 128:, :, :]              # skip2 梯度 (N, 64, 16, 16)

    # --- skip2 拆分 ---
    dx_skip2_feat = dx_skip2[:, :32, :, :]            # 特征部分 (N, 32, 16, 16)
    dx_time_enc2_from_skip = dx_skip2[:, 32:, :, :].sum(axis=(2,3))  # (N, 32)

    # --- up3 反向 ---
    N, C, H, W = dx_up3.shape   # (N,128,16,16)
    dx_upsampled = np.zeros((N, C, H//2, W//2), dtype=dx_up3.dtype)
    for i in range(2):
        for j in range(2):
            dx_upsampled += dx_up3[:, :, i::2, j::2]
    dx_bn_out = dx_upsampled  # (N, 128, 8, 8)

    # bn_out 结构：特征64 + 时间64
    dx_bn_feat = dx_bn_out[:, :64, :, :]              # 特征部分 (N, 64, 8, 8)
    dx_time_bn = dx_bn_out[:, 64:, :, :].sum(axis=(2,3))  # (N, 64)
    caches['dx_time_bn'] = dx_time_bn

    dx = dx_bn_feat * d_relu(None, caches['relu_bn_before_time'])

    dx_before_bn, dgamma, dbeta = d_batch_norm(dx, caches['bn_bn'])
    grads['bn_bn_gamma'] = dgamma
    grads['bn_bn_beta'] = dbeta
    dx = dx_before_bn
    # ---------- 瓶颈 (bn) ----------
    # 输入 bn_input = enc3_out (128) ，结构：特征64 + 时间64
    bn_input, w, b, stride, padding, x_pad, H_out, W_out = caches['bn']
    dx_bn_input, dw_bn, db_bn = d_conv(dx, bn_input, w, stride, padding, x_pad, H_out, W_out)
    grads['bn_w'] = dw_bn
    grads['bn_b'] = db_bn

    # 拆分 bn_input 梯度：前64为 enc3 特征，后64为时间
    dx_enc3_feat = dx_bn_input[:, :64, :, :]          # 特征部分 (N, 64, 8, 8)
    dx_time_enc3 = dx_bn_input[:, 64:, :, :].sum(axis=(2,3))  # (N, 64)
    caches['dx_time_enc3'] = dx_time_enc3

    dx = dx_enc3_feat * d_relu(None, caches['relu_enc3_before_time'])
    dx_before_bn, dgamma, dbeta = d_batch_norm(dx, caches['bn_enc3'])
    grads['bn_enc3_gamma'] = dgamma
    grads['bn_enc3_beta'] = dbeta
    dx = dx_before_bn

    # ---------- Enc3 ----------
    # 输入 enc3_input = enc2_out (64) ，结构：特征32 + 时间32
    enc3_input, w, b, stride, padding, x_pad, H_out, W_out = caches['enc3']
    dx_enc3_input, dw_enc3, db_enc3 = d_conv(dx, enc3_input, w, stride, padding, x_pad, H_out, W_out)
    grads['enc3_w'] = dw_enc3
    grads['enc3_b'] = db_enc3

    # 拆分 enc3_input 梯度
    dx_enc2_feat = dx_enc3_input[:, :32, :, :]        # 特征部分 (N, 32, 16, 16)
    dx_time_enc2_from_enc3 = dx_enc3_input[:, 32:, :, :].sum(axis=(2,3))  # (N, 32)

    # 累加来自 skip2 和 enc3 的时间梯度
    caches['dx_time_enc2'] = dx_time_enc2_from_skip + dx_time_enc2_from_enc3

    # 特征部分通过 ReLU 导数 (relu_enc2_before_time 是 enc2 卷积输出 relu 后)
    dx = dx_enc2_feat * d_relu(None, caches['relu_enc2_before_time'])

    dx_before_bn, dgamma, dbeta = d_batch_norm(dx, caches['bn_enc2'])
    grads['bn_enc2_gamma'] = dgamma
    grads['bn_enc2_beta'] = dbeta
    dx = dx_before_bn
    # ---------- Enc2 ----------
    # 输入 enc2_input = enc1_out (32) ，结构：特征16 + 时间16
    enc2_input, w, b, stride, padding, x_pad, H_out, W_out = caches['enc2']
    dx_enc2_input, dw_enc2, db_enc2 = d_conv(dx, enc2_input, w, stride, padding, x_pad, H_out, W_out)
    grads['enc2_w'] = dw_enc2
    grads['enc2_b'] = db_enc2

    dx_enc1_feat = dx_enc2_input[:, :16, :, :]        # 特征部分 (N, 16, 32, 32)
    dx_time_enc1_from_enc2 = dx_enc2_input[:, 16:, :, :].sum(axis=(2,3))  # (N, 16)

    # 累加来自 skip1 和 enc2 的时间梯度
    caches['dx_time_enc1'] = dx_time_enc1_from_skip + dx_time_enc1_from_enc2

    # 特征部分还要加上来自 skip1 的特征梯度（已经计算过 dx_skip1_feat）
    dx_enc1_feat += dx_skip1_feat

    dx = dx_enc1_feat * d_relu(None, caches['relu_enc1_before_time'])
    enc3_input, w, b, stride, padding, x_pad, H_out, W_out = caches['enc1']
    dx_enc3_input, dw_enc3, db_enc3 = d_conv(dx, enc3_input, w, stride, padding, x_pad, H_out, W_out)
    grads['enc1_w'] = dw_enc3
    grads['enc1_b'] = db_enc3

    # ---------- Enc1 ----------
    # 输入 x (1通道) ，无时间
    enc1_input, w, b, stride, padding, x_pad, H_out, W_out = caches['enc1']
    _, dw_enc1, db_enc1 = d_conv(dx, enc1_input, w, stride, padding, x_pad, H_out, W_out)
    grads['enc1_w'] = dw_enc1
    grads['enc1_b'] = db_enc1

    # ---------- 计算时间 FC 层梯度 ----------
    time_emb = caches['time_emb']  # (N, 64)

    def compute_fc_grad(d_time, name):
        # d_time: (N, out_dim) 已经过空间求和
        # 获取该层时间特征 ReLU 前的输出（用于掩码）
        time_raw = caches[f'time_feat_raw_{name}']   # (N, out_dim)
        mask = (time_raw > 0).astype(np.float32)    # (N, out_dim)
        d_time = d_time * mask                       # 应用 ReLU 导数
        w_key = f'time_fc_{name}_w'
        b_key = f'time_fc_{name}_b'
        dw = time_emb.T @ d_time                     # (64, out_dim)
        db = d_time.sum(axis=0)                      # (out_dim,)
        grads[w_key] = dw
        grads[b_key] = db

    # 每个层的时间梯度已存入 caches['dx_time_*']
    compute_fc_grad(caches['dx_time_enc1'], 'enc1')
    compute_fc_grad(caches['dx_time_enc2'], 'enc2')
    compute_fc_grad(caches['dx_time_enc3'], 'enc3')
    compute_fc_grad(caches['dx_time_bn'], 'bn')
    compute_fc_grad(caches['dx_time_dec3'], 'dec3')
    compute_fc_grad(caches['dx_time_dec2'], 'dec2')

    return grads
# ============================================================
# 4. 数据加载
# ============================================================
def load_mnist_single_padding(pkl_path='/Users/zhaomingming/data_sets/mnist/mnist.pkl'):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    (X_train, y_train), _, _ = data
    if X_train.ndim == 4:
        X = X_train[0, 0]
    elif X_train.ndim == 3:
        X = X_train[0]
    else:
        X = X_train[0]
    X = X.astype(np.float32) / 127.5 -1.0
    X = X.reshape(1, 1, 28, 28)
    X_pad = np.zeros((1, 1, 32, 32), dtype=np.float32) - 1.0
    X_pad[:, :, 2:30, 2:30] = X
    return X_pad
import numpy as np

def bilinear_resize(img, out_shape):
    """
    img: (H, W) float32，范围任意（通常是 [0,1] 或 [-1,1]）
    out_shape: (out_h, out_w)
    返回: (out_h, out_w) float32，数值范围与输入相同
    """
    in_h, in_w = img.shape
    out_h, out_w = out_shape
    
    # 计算输出像素在输入图像中的对应坐标（中心对齐）
    scale_h = in_h / out_h
    scale_w = in_w / out_w
    
    # 生成输出网格坐标
    out_y = np.arange(out_h, dtype=np.float32) * scale_h + 0.5 * (scale_h - 1)
    out_x = np.arange(out_w, dtype=np.float32) * scale_w + 0.5 * (scale_w - 1)
    
    # 找出四个最近邻像素的索引
    y0 = np.floor(out_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x0 = np.floor(out_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, in_w - 1)
    
    # 计算插值权重
    dy = out_y - y0
    dx = out_x - x0
    
    # 广播形状以便向量化
    # 最终输出 (out_h, out_w)
    y0 = y0[:, None]
    y1 = y1[:, None]
    dy = dy[:, None]
    
    # 获取四个角像素值
    I00 = img[y0, x0]   # (out_h, out_w)
    I01 = img[y0, x1]
    I10 = img[y1, x0]
    I11 = img[y1, x1]
    
    # 双线性插值
    top = I00 + (I01 - I00) * dx
    bottom = I10 + (I11 - I10) * dx
    result = top + (bottom - top) * dy
    
    return result
def load_mnist_single(pkl_path='/Users/zhaomingming/data_sets/mnist/mnist.pkl'):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    (X_train, y_train), _, _ = data
    if X_train.ndim == 4:
        X = X_train[0, 0]
    elif X_train.ndim == 3:
        X = X_train[0]
    else:
        X = X_train[0]
    # 缩放到 32x32（双线性插值）
    img_resized = bilinear_resize(X,(32, 32))
    # 归一化到 [-1, 1]
    X_resized = img_resized * 2.0 - 1.0
    # 添加 batch 和 channel 维度
    X_resized = X_resized.reshape(1, 1, 32, 32)
    return X_resized

# ------------------------------------------------------------
# 2. RGB转灰度（纯 NumPy）
# ------------------------------------------------------------
def rgb_to_grayscale(rgb_img):
    """
    rgb_img: (H, W, 3) float32 或 uint8，范围 [0,255] 或 [0,1]
    返回: (H, W) float32，范围保持与输入一致（即若输入 [0,255]，输出 [0,255]）
    """
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError("输入必须是 (H, W, 3) 的 RGB 图像")
    # 统一转为 float32 并确定最大值
    if rgb_img.dtype == np.uint8:
        rgb_img = rgb_img.astype(np.float32)
        max_val = 255.0
    else:
        max_val = 1.0 if rgb_img.max() <= 1.0 else 255.0
    # 标准灰度权重
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.dot(rgb_img[..., :3], weights)   # (H, W)
    # 保证数值范围不变（若输入为0-255，输出也为0-255）
    return gray

# ------------------------------------------------------------
# 3. 加载单张图片（从文件或文件夹）
# ------------------------------------------------------------
def load_single_image(path, target_size=(32, 32)):
    """
    加载一张彩色图片，转为灰度，缩放到 target_size，归一化到 [-1,1]。
    - path: 图片文件路径或文件夹路径（随机选一张）
    - target_size: (H, W)
    返回: (1, 1, H, W) float32，范围 [-1,1]
    """
    # 处理文件夹
    if os.path.isdir(path):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(path, e)))
        if not files:
            raise ValueError(f"在文件夹 {path} 中未找到图片")
        img_path = np.random.choice(files)
        print(f"随机选择图片: {img_path}")
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        img_path = path

    # 读取图片（使用 PIL，可替换为 cv2.imread）
    img = Image.open(img_path).convert('RGB')   # 确保 RGB
    rgb = np.array(img, dtype=np.uint8)         # (H, W, 3) uint8

    # 转灰度
    gray = rgb_to_grayscale(rgb)                # (H, W) float32, 范围 0-255

    # 缩放到目标尺寸
    gray_resized = bilinear_resize(gray, target_size)   # (H, W) float32

    # 归一化到 [-1,1]
    gray_norm = (gray_resized / 127.5) - 1.0

    # 添加 batch 和 channel 维度
    gray_norm = gray_norm.reshape(1, 1, target_size[0], target_size[1])
    return gray_norm

# ------------------------------------------------------------
# 可选：调试保存（纯 NumPy + PIL 保存）
# ------------------------------------------------------------
def save_debug_image(arr, save_path='debug.png'):
    """保存 (1,1,H,W) 到 PNG，范围 [-1,1] 自动映射到 0-255"""
    img = ((arr[0, 0] + 1.0) * 127.5).astype(np.uint8)
    Image.fromarray(img, mode='L').save(save_path)
    print(f"调试图像已保存: {save_path}")
# ============================================================
# 5. 训练 & 采样
# ============================================================
def train_single_full():
    np.random.seed(67)
    timesteps =100 
    epochs = 100*800
    lr = 0.01
#x0 = load_mnist_single()
    x0 = load_single_image('./imgs/')
    betas = linear_beta_schedule(timesteps)
    sqrt_alphas_bar, sqrt_one_minus_alphas_bar = get_sqrt_alphas_bar(betas)
    weights = init_diffusion_weights()
    # 将所有权重设为0
#for k in weights:
#        weights[k] = np.zeros_like(weights[k])*100.0
#momentum = 0.9
#velocity = {k: np.zeros_like(v) for k, v in weights.items()}
    # 初始化 Adam 一阶矩（m）和二阶矩（v），均为零
    m = {k: np.zeros_like(v) for k, v in weights.items()}
    v = {k: np.zeros_like(v) for k, v in weights.items()}

    # Adam 超参数（常用默认值）
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    best_path = 'best_xing_model.npz'
    latest_path = 'latest_xing_checkpoint.npz'
# 尝试加载最佳模型
    loaded_w, loaded_m, loaded_v, loaded_epoch, loaded_losses = None, None, None, 0, []
    if os.path.exists(best_path):
        loaded_w, loaded_m, loaded_v, loaded_epoch, loaded_losses = load_checkpoint(best_path)
        print(f"✅ 从最佳模型恢复，起始 epoch: {loaded_epoch+1}, 历史最佳损失: {min(loaded_losses) if loaded_losses else 'N/A'}")
    elif os.path.exists(latest_path):
        loaded_w, loaded_m, loaded_v, loaded_epoch, loaded_losses = load_checkpoint(latest_path)
        print(f"✅ 从最新检查点恢复，起始 epoch: {loaded_epoch+1}")
    if loaded_w is not None:
        weights = loaded_w
        m = loaded_m if loaded_m is not None else {k: np.zeros_like(v) for k, v in weights.items()}
        v = loaded_v if loaded_v is not None else {k: np.zeros_like(v) for k, v in weights.items()}
        start_epoch = loaded_epoch
        losses = loaded_losses
        best_loss = min(losses) if losses else float('inf')
    else:
        weights = init_diffusion_weights()
        m = {k: np.zeros_like(v) for k, v in weights.items()}
        v = {k: np.zeros_like(v) for k, v in weights.items()}
        start_epoch = 0
        losses = []
        best_loss = float('inf')


    losses = []
    accum_steps = 4
    for epoch in range(epochs):
#        lr = cosine_annealing_lr(epoch, epochs, lr_max=0.01, lr_min=1e-6)
        lr = cosine_annealing_lr(epoch, epochs, lr_max=0.00001, lr_min=0.000001)
        # 然后用这个 lr 更新权重（对于 Adam，lr 即为学习率）
		  # 清零累积梯度
        grads_accum = {k: np.zeros_like(v) for k, v in weights.items()}
        for i in range(accum_steps):
            t = np.random.randint(0, timesteps, size=1)
            noise = np.random.randn(1, 1, 32, 32).astype(np.float32)
            sqrt_alpha_bar_t = sqrt_alphas_bar[t][0]
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_bar[t][0]
            x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
            '''
            if (epoch + 1) % 1 == 0:
            print(f"\nEpoch {epoch+1} weights abs means:")
            for key in weights:
                mean_abs = np.abs(weights[key]).mean()
                print(f"  {key}: {mean_abs:.6f}")
		    '''
            pred_noise, caches = forward_diffusion(x_t, t, weights)
            '''
            # ---- 检查 ReLU 输出是否死亡 ----
            if (epoch + 1) % 1 == 0:   # 或 epoch % 50 == 0
                print(f"\nEpoch {epoch+1} ReLU statistics:")
                for key in caches:
                    if 'relu' in key and 'before_time' not in key:  # 只检查 ReLU 后的激活（不含拼接后的）
                        val = caches[key]
                        pos_ratio = (val > 0).mean()
                        print(f"  {key}: shape={val.shape}, mean={val.mean():.6f}, std={val.std():.6f}, pos_ratio={pos_ratio:.4f}")
	        '''
            loss = np.mean((pred_noise - noise) ** 2)
            losses.append(loss)
            dloss = 2.0 * (pred_noise - noise) / pred_noise.size
            grads = backward_diffusion(dloss, caches, weights)
		    # 累加梯度
            for key in grads:
                grads_accum[key] += grads[key]
        # ---- 如果当前损失更低，保存最佳模型 ----
        if loss < best_loss:
            best_loss = loss
            # 最佳模型只保存权重、epoch 和损失（可以不保存优化器状态，但为方便继续训练，也保存）
            save_checkpoint(best_path, weights, m, v, epoch+1, losses, 'adam')
            print(f"⭐ 新的最佳模型已保存，epoch={epoch+1}, loss={loss:.6f}")

#        for key in grads:
#            grads[key] = np.clip(grads[key], -1.0, 1.0)
#            velocity[key] = momentum * velocity[key] - lr * grads[key]
#            weights[key] += velocity[key]
#            weights[key] += -lr * grads[key]
        # 平均梯度（可选）或直接用累加值更新
        for key in weights:
            grads_accum[key] /= accum_steps   # 取平均，保持更新尺度一致
            # Adam 或 SGD 更新
            m[key] = beta1 * m[key] + (1 - beta1) * grads_accum[key]
            v[key] = beta2 * v[key] + (1 - beta2) * (grads_accum[key] ** 2)
            weights[key] -= lr * m[key] / (np.sqrt(v[key]) + eps)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
            '''
            print(f"\nEpoch {epoch+1} gradient abs means:")
            for key in grads:
                mean_abs = np.abs(grads[key]).mean()
                print(f"  {key}: {mean_abs:.6f}")
            '''

    return weights, betas


def sample_diffusion(weights, betas, n_samples=1):
    timesteps = len(betas)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)

    x_t = np.random.randn(n_samples, 1, 32, 32).astype(np.float32)

    for t in range(timesteps - 1, -1, -1):
        t_batch = np.full((n_samples,), t, dtype=np.int32)
        pred_noise, _ = forward_diffusion(x_t, t_batch, weights)
        alpha_t = alphas[t]
        beta_t = betas[t]
        alpha_bar_t = alphas_bar[t]
        x_t = (1.0 / np.sqrt(alpha_t)) * (x_t - beta_t / np.sqrt(1.0 - alpha_bar_t) * pred_noise)
        if t > 0:
            z = np.random.randn(*x_t.shape).astype(np.float32)
            x_t = x_t + np.sqrt(beta_t) * z

    return x_t


def main():
    weights, betas = train_single_full()
    np.savez('diffusion_full.npz', **weights)
    print("权重已保存")

    gen = sample_diffusion(weights, betas, n_samples=1)
    plt.figure()
    plt.imshow(gen[0, 0], cmap='gray')
    plt.title('Generated Full Model')
    plt.axis('off')
    plt.savefig('generated_full.png')
    plt.show()


if __name__ == "__main__":
   main()
