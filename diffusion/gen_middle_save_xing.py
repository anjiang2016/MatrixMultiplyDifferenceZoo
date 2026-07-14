import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import os
from train_diffusion_single_stand import forward_diffusion, init_diffusion_weights, get_timestep_embedding, upsample_nearest, batch_norm,  conv, linear, relu, d_conv, d_relu, d_linear, d_batch_norm
import pickle
# 但 forward_diffusion 只需要前向函数，不需要反向函数，所以只导入 forward 相关的即可

# 假设你的 forward_diffusion 函数和权重初始化函数在同一文件中，或者你已导入
# 这里为了独立，我们假设你已经有 forward_diffusion 和 init_diffusion_weights 的定义
# 如果脚本独立，请将 forward_diffusion 和相关的函数（如 get_timestep_embedding, upsample_nearest, batch_norm等）复制到此处。

def load_best_model(best_path='best_xing_model.npz'):
    """加载最佳模型权重"""
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"最佳模型文件 {best_path} 不存在，请先训练。")
    data = np.load(best_path, allow_pickle=True)
    weights = data['weights'].item()
    print(f"✅ 加载最佳模型，epoch={data['epoch']}, loss={min(data['losses']):.6f}")
    return weights

def linear_beta_schedule(timesteps=100):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)

def sample_diffusion_with_save(weights, betas, n_samples=1, save_interval=10, save_dir='sampling_steps'):
    """采样并每隔 save_interval 步保存中间图像"""
    os.makedirs(save_dir, exist_ok=True)
    
    timesteps = len(betas)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    
    x_t = np.random.randn(n_samples, 1, 32, 32).astype(np.float32)
    
    # 保存初始噪声
    save_image(x_t[0, 0], save_dir, f'step_{timesteps}_init')
    
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
        
        # 每隔 save_interval 步保存当前图像（包括 t=0）
        if t % save_interval == 0:
            save_image(x_t[0, 0], save_dir, f'step_{t}')
    
    # 最后再保存一次最终结果（t=0 已经保存，但为了清晰）
    save_image(x_t[0, 0], save_dir, 'final')
    print("Final x_t: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(x_t.mean(), x_t.std(), x_t.min(), x_t.max()))
    return x_t

def save_image(img_array, save_dir, name):
    """将 (32,32) 的 float 数组映射到 [0,255] 并保存为 PNG"""
    # 假设训练数据归一化为 [-1, 1]
    img = (img_array + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img, mode='L')
    img_pil.save(os.path.join(save_dir, f'{name}.png'))
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

betas = linear_beta_schedule(100)   # 你的 betas
weights = load_best_model('best_xing_model.npz')

# 采样并保存中间图像
x_final = sample_diffusion_with_save(weights, betas, n_samples=1, save_interval=10, save_dir='sampling_xing_steps')
