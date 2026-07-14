"""
train_diffusion_single_timestep.py
单张图片过拟合 + 时间步嵌入 + 多步采样
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funcs import conv, d_conv, relu, d_relu, linear, d_linear


# ============================================================
# 1. 时间嵌入
# ============================================================
def get_timestep_embedding(t, emb_dim=64):
    half_dim = emb_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    return emb


# ============================================================
# 2. 噪声调度
# ============================================================
def linear_beta_schedule(timesteps=100):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)


def get_sqrt_alphas_bar(betas):
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    return np.sqrt(alphas_bar), np.sqrt(1.0 - alphas_bar)


# ============================================================
# 3. 模型定义（带时间嵌入）
# ============================================================
def init_diffusion_weights():
    weights = {}

    def he_init(shape):
        fan_in = np.prod(shape[1:]) if len(shape) > 1 else shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape).astype(np.float32) * std

    # 编码器
    weights['enc1_w'] = he_init((16, 1, 3, 3))
    weights['enc1_b'] = np.zeros(16, dtype=np.float32)
    weights['enc2_w'] = he_init((32, 16, 3, 3))
    weights['enc2_b'] = np.zeros(32, dtype=np.float32)
    weights['enc3_w'] = he_init((64, 32, 3, 3))
    weights['enc3_b'] = np.zeros(64, dtype=np.float32)

    # 时间嵌入 FC
    weights['time_fc_w'] = he_init((64, 64))
    weights['time_fc_b'] = np.zeros(64, dtype=np.float32)

    # 瓶颈
    weights['bn_w'] = he_init((64, 64, 3, 3))
    weights['bn_b'] = np.zeros(64, dtype=np.float32)

    # 解码器
    weights['dec3_w'] = he_init((32, 64+32, 3, 3))
    weights['dec3_b'] = np.zeros(32, dtype=np.float32)
    weights['dec2_w'] = he_init((16, 32+16, 3, 3))
    weights['dec2_b'] = np.zeros(16, dtype=np.float32)
    weights['dec1_w'] = he_init((1, 16+1, 3, 3))
    weights['dec1_b'] = np.zeros(1, dtype=np.float32)

    return weights


def upsample_nearest(x, scale_factor=2):
    N, C, H, W = x.shape
    return x.repeat(scale_factor, axis=2).repeat(scale_factor, axis=3)


def forward_diffusion(x, t, weights):
    caches = {}

    # 时间嵌入
    time_emb = get_timestep_embedding(t, 64)
    time_feat, _ = linear(time_emb, weights['time_fc_w'], weights['time_fc_b'])
    caches['time_emb'] = time_emb
    caches['time_feat'] = time_feat
    time_feat_relu = relu(time_feat)
    caches['time_feat_relu'] = time_feat_relu

    # Enc1
    out, x_pad, (H_out, W_out) = conv(x, weights['enc1_w'], weights['enc1_b'], stride=1, padding=1)
    caches['enc1'] = (x, weights['enc1_w'], weights['enc1_b'], 1, 1, x_pad, H_out, W_out)
    out = relu(out)
    skip1 = out
    caches['skip1'] = skip1
    caches['relu1'] = out

    # Enc2
    conv2_input = out
    out, x_pad, (H_out, W_out) = conv(conv2_input, weights['enc2_w'], weights['enc2_b'], stride=2, padding=1)
    caches['enc2'] = (conv2_input, weights['enc2_w'], weights['enc2_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    skip2 = out
    caches['skip2'] = skip2
    caches['relu2'] = out

    # Enc3
    conv3_input = out
    out, x_pad, (H_out, W_out) = conv(conv3_input, weights['enc3_w'], weights['enc3_b'], stride=2, padding=1)
    caches['enc3'] = (conv3_input, weights['enc3_w'], weights['enc3_b'], 2, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu3'] = out

    # 瓶颈 + 时间
    time_exp = time_feat_relu[:, :, None, None]
    out = out + time_exp
    caches['bn_input'] = out
    bn_input = out
    out, x_pad, (H_out, W_out) = conv(bn_input, weights['bn_w'], weights['bn_b'], stride=1, padding=1)
    caches['bn'] = (bn_input, weights['bn_w'], weights['bn_b'], 1, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu_bn'] = out

    # Dec3
    up3 = upsample_nearest(out, 2)
    concat3 = np.concatenate([up3, skip2], axis=1)
    caches['concat3'] = concat3
    dec3_input = concat3
    out, x_pad, (H_out, W_out) = conv(dec3_input, weights['dec3_w'], weights['dec3_b'], stride=1, padding=1)
    caches['dec3'] = (dec3_input, weights['dec3_w'], weights['dec3_b'], 1, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu_dec3'] = out

    # Dec2
    up2 = upsample_nearest(out, 2)
    concat2 = np.concatenate([up2, skip1], axis=1)
    caches['concat2'] = concat2
    dec2_input = concat2
    out, x_pad, (H_out, W_out) = conv(dec2_input, weights['dec2_w'], weights['dec2_b'], stride=1, padding=1)
    caches['dec2'] = (dec2_input, weights['dec2_w'], weights['dec2_b'], 1, 1, x_pad, H_out, W_out)
    out = relu(out)
    caches['relu_dec2'] = out

    # Dec1
    concat1 = np.concatenate([out, x], axis=1)
    caches['concat1'] = concat1
    dec1_input = concat1
    out, x_pad, (H_out, W_out) = conv(dec1_input, weights['dec1_w'], weights['dec1_b'], stride=1, padding=1)
    caches['dec1'] = (dec1_input, weights['dec1_w'], weights['dec1_b'], 1, 1, x_pad, H_out, W_out)

    return out, caches


def backward_diffusion(dout, caches, weights):
    grads = {}
    dx = dout

    # Dec1
    dec1_input, w, b, stride, padding, x_pad, H_out, W_out = caches['dec1']
    dx, dw, db = d_conv(dx, dec1_input, w, stride, padding, x_pad, H_out, W_out)
    grads['dec1_w'] = dw
    grads['dec1_b'] = db
    dx_conv_dec2 = dx[:, :16, :, :]
    dx = dx_conv_dec2 * d_relu(None, caches['relu_dec2'])

    # Dec2
    dec2_input, w, b, stride, padding, x_pad, H_out, W_out = caches['dec2']
    dx, dw, db = d_conv(dx, dec2_input, w, stride, padding, x_pad, H_out, W_out)
    grads['dec2_w'] = dw
    grads['dec2_b'] = db
    dx_upsample = dx[:, :32, :, :]
    N, C, H, W = dx_upsample.shape
    dx_upsampled = np.zeros((N, C, H//2, W//2), dtype=dx.dtype)
    for i in range(2):
        for j in range(2):
            dx_upsampled += dx_upsample[:, :, i::2, j::2]
    dx = dx_upsampled / 4.0
    dx = dx * d_relu(None, caches['relu_dec3'])

    # Dec3
    dec3_input, w, b, stride, padding, x_pad, H_out, W_out = caches['dec3']
    dx, dw, db = d_conv(dx, dec3_input, w, stride, padding, x_pad, H_out, W_out)
    grads['dec3_w'] = dw
    grads['dec3_b'] = db
    dx_upsample = dx[:, :64, :, :]
    N, C, H, W = dx_upsample.shape
    dx_upsampled = np.zeros((N, C, H//2, W//2), dtype=dx.dtype)
    for i in range(2):
        for j in range(2):
            dx_upsampled += dx_upsample[:, :, i::2, j::2]
    dx = dx_upsampled / 4.0
    dx = dx * d_relu(None, caches['relu_bn'])

    # 瓶颈
    bn_input, w, b, stride, padding, x_pad, H_out, W_out = caches['bn']
    dx, dw, db = d_conv(dx, bn_input, w, stride, padding, x_pad, H_out, W_out)
    grads['bn_w'] = dw
    grads['bn_b'] = db

    # 时间嵌入反向
    d_time_feat = dx.sum(axis=(2, 3))
    time_feat = caches['time_feat']
    time_feat_relu = caches['time_feat_relu']
    d_time_feat = d_time_feat * d_relu(time_feat, time_feat_relu)
    time_emb = caches['time_emb']
    w_fc = weights['time_fc_w']
    dw_fc = time_emb.T @ d_time_feat
    db_fc = d_time_feat.sum(axis=0)
    grads['time_fc_w'] = dw_fc
    grads['time_fc_b'] = db_fc

    # 编码器反向
    dx_enc3 = dx
    dx = dx_enc3 * d_relu(None, caches['relu3'])
    enc3_input, w, b, stride, padding, x_pad, H_out, W_out = caches['enc3']
    dx, dw, db = d_conv(dx, enc3_input, w, stride, padding, x_pad, H_out, W_out)
    grads['enc3_w'] = dw
    grads['enc3_b'] = db
    dx = dx * d_relu(None, caches['relu2'])
    enc2_input, w, b, stride, padding, x_pad, H_out, W_out = caches['enc2']
    dx, dw, db = d_conv(dx, enc2_input, w, stride, padding, x_pad, H_out, W_out)
    grads['enc2_w'] = dw
    grads['enc2_b'] = db
    dx = dx * d_relu(None, caches['relu1'])
    enc1_input, w, b, stride, padding, x_pad, H_out, W_out = caches['enc1']
    dx, dw, db = d_conv(dx, enc1_input, w, stride, padding, x_pad, H_out, W_out)
    grads['enc1_w'] = dw
    grads['enc1_b'] = db

    return grads


# ============================================================
# 4. 数据加载（单张图片）
# ============================================================
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
    X = X.astype(np.float32) / 255.0
    X = X.reshape(1, 1, 28, 28)
    X_pad = np.zeros((1, 1, 32, 32), dtype=np.float32)
    X_pad[:, :, 2:30, 2:30] = X
    return X_pad


# ============================================================
# 5. 训练（单张图片，随机时间步）
# ============================================================
def train_single_timestep():
    np.random.seed(42)

    timesteps = 100
    epochs = 1000
    lr = 0.001

    x0 = load_mnist_single()
    print(f"原始图像形状: {x0.shape}")

    betas = linear_beta_schedule(timesteps)
    sqrt_alphas_bar, sqrt_one_minus_alphas_bar = get_sqrt_alphas_bar(betas)

    weights = init_diffusion_weights()

    losses = []
    for epoch in range(epochs):
        # 随机时间步
        t = np.random.randint(0, timesteps, size=1)

        # 加噪
        noise = np.random.randn(1, 1, 32, 32).astype(np.float32)
        sqrt_alpha_bar_t = sqrt_alphas_bar[t][0]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_bar[t][0]
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        # 预测噪声
        pred_noise, caches = forward_diffusion(x_t, t, weights)

        # Loss
        loss = np.mean((pred_noise - noise) ** 2)
        losses.append(loss)

        # 反向
        dloss = 2.0 * (pred_noise - noise) / pred_noise.size
        grads = backward_diffusion(dloss, caches, weights)

        for key in grads:
            grads[key] = np.clip(grads[key], -1.0, 1.0)
            weights[key] -= lr * grads[key]

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.6f}")

    return weights, betas


# ============================================================
# 6. 多步采样
# ============================================================
def sample_diffusion(weights, betas, n_samples=1):
    timesteps = len(betas)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    sqrt_alphas_bar = np.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = np.sqrt(1.0 - alphas_bar)

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


# ============================================================
# 7. 主函数
# ============================================================
def main():
    weights, betas = train_single_timestep()
    np.savez('diffusion_single_timestep.npz', **weights)
    print("权重已保存")

    # 生成一张新图片
    gen = sample_diffusion(weights, betas, n_samples=1)

    plt.figure()
    plt.imshow(gen[0, 0], cmap='gray')
    plt.title('Generated (Single image timestep)')
    plt.axis('off')
    plt.savefig('generated_single_timestep.png')
    plt.show()


if __name__ == "__main__":
    main()
