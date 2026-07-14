import numpy as np
import matplotlib.pyplot as plt
import os
from train_diffusion_single_stand import forward_diffusion, init_diffusion_weights, get_timestep_embedding, upsample_nearest, batch_norm,  conv, linear, relu, d_conv, d_relu, d_linear, d_batch_norm
# 但 forward_diffusion 只需要前向函数，不需要反向函数，所以只导入 forward 相关的即可

# 假设你的 forward_diffusion 函数和权重初始化函数在同一文件中，或者你已导入
# 这里为了独立，我们假设你已经有 forward_diffusion 和 init_diffusion_weights 的定义
# 如果脚本独立，请将 forward_diffusion 和相关的函数（如 get_timestep_embedding, upsample_nearest, batch_norm等）复制到此处。

def load_best_model(best_path='best_model.npz'):
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

def sample_diffusion(weights, betas, n_samples=1):
    """从纯噪声生成图像（DDPM采样）"""
    timesteps = len(betas)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    
    # 从标准正态分布采样初始噪声
    x_t = np.random.randn(n_samples, 1, 32, 32).astype(np.float32)
    
    for t in range(timesteps - 1, -1, -1):
        t_batch = np.full((n_samples,), t, dtype=np.int32)
        # 预测当前噪声
        pred_noise, _ = forward_diffusion(x_t, t_batch, weights)
        if t == timesteps - 1:
            print(f"Initial x_t mean={x_t.mean():.4f}, std={x_t.std():.4f}")
            print(f"pred_noise mean={pred_noise.mean():.4f}, std={pred_noise.std():.4f}")
        alpha_t = alphas[t]
        beta_t = betas[t]
        alpha_bar_t = alphas_bar[t]
        # 去噪公式
        x_t = (1.0 / np.sqrt(alpha_t)) * (x_t - beta_t / np.sqrt(1.0 - alpha_bar_t) * pred_noise)
        if t % 10 == 0:  # 每10步打印一次
             print(f"t={t}, x_t mean={x_t.mean():.4f}, std={x_t.std():.4f}, min={x_t.min():.4f}, max={x_t.max():.4f}")
        if t > 0:
            z = np.random.randn(*x_t.shape).astype(np.float32)
            x_t = x_t + np.sqrt(beta_t) * z
    return x_t

def main():
    # 设置路径
    best_model_path = 'best_model.npz'   # 你保存的最佳模型文件
    timesteps = 100                      # 必须与训练时一致
    
    # 加载权重
    weights = load_best_model(best_model_path)
    
    # 生成 beta schedule（必须与训练时相同）
    betas = linear_beta_schedule(timesteps)
    
    # 采样生成图像
    gen = sample_diffusion(weights, betas, n_samples=1)
    
    # 将输出从 [-1, 1] 映射到 [0, 255]
    gen_img = (gen[0, 0] + 1.0) * 127.5
    gen_img = np.clip(gen_img, 0, 255).astype(np.uint8)
    
    # 显示并保存
    plt.figure(figsize=(4, 4))
    plt.imshow(gen_img, cmap='gray')
    plt.title('Generated Image (T=100)')
    plt.axis('off')
    plt.savefig('generated_image.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    print("✅ 生成图像已保存为 generated_image.png")

if __name__ == "__main__":
    main()
