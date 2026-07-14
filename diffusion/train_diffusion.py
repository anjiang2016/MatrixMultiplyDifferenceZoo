"""
train_diffusion.py - 训练扩散模型
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from diffusion import init_diffusion_weights, forward_diffusion
from funcs import conv, d_conv, relu, d_relu, linear, d_linear

def linear_beta_schedule(timesteps=1000):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, sqrt_alphas_bar, sqrt_one_minus_alphas_bar):
    """前向加噪"""
    noise = np.random.randn(*x_start.shape).astype(np.float32)
    sqrt_alphas_bar_t = sqrt_alphas_bar[t][:, None, None, None]
    sqrt_one_minus_alphas_bar_t = sqrt_one_minus_alphas_bar[t][:, None, None, None]
    x_t = sqrt_alphas_bar_t * x_start + sqrt_one_minus_alphas_bar_t * noise
    return x_t, noise

def load_mnist_simple(num_samples=128):
    """加载 MNIST 子集"""
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), _ = mnist.load_data()
    X_train = X_train[:num_samples]
    X_train = X_train.astype(np.float32) / 255.0
    X_train = X_train.reshape(-1, 1, 28, 28)
    # Pad 到 32x32
    X_pad = np.zeros((num_samples, 1, 32, 32), dtype=np.float32)
    X_pad[:, :, 2:30, 2:30] = X_train
    return X_pad

def train_diffusion():
    # 超参数
    timesteps = 100
    num_samples = 128
    batch_size = 16
    epochs = 10
    lr = 0.001
    
    # 数据
    data = load_mnist_simple(num_samples)
    print(f"数据形状: {data.shape}")
    
    # 噪声调度
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    sqrt_alphas_bar = np.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = np.sqrt(1.0 - alphas_bar)
    
    # 模型
    weights = init_diffusion_weights()
    
    print("开始训练扩散模型...")
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        total_loss = 0.0
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = data[batch_idx]
            
            # 随机时间步
            t = np.random.randint(0, timesteps, size=len(batch_idx))
            
            # 加噪
            x_t, noise = q_sample(batch_x, t, sqrt_alphas_bar, sqrt_one_minus_alphas_bar)
            
            # 预测噪声
            pred_noise, caches = forward_diffusion(x_t, t, weights)
            
            # MSE 损失
            loss = np.mean((pred_noise - noise) ** 2)
            total_loss += loss
            
            # 反向传播（需要手写或使用自动微分）
            # 这里简化，只做前向演示
            # 实际训练需要实现 backward_diffusion
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(num_samples//batch_size):.6f}")
    
    return weights

if __name__ == "__main__":
    weights = train_diffusion()
