import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def print_params_l1(params, prefix='', filter_keys=None, total_only=False):
    """
    递归遍历参数字典，打印每个数组的 L1 范数（绝对值之和）。

    参数:
        params: 嵌套字典，叶子节点为 numpy 数组
        prefix: 当前层级前缀（用于打印）
        filter_keys: 可选，字符串元组或列表，只打印包含这些子串的键（如 ('_w','_b')）
        total_only: 若 True，只打印总计不打印单项
    返回:
        total_l1: 所有数组 L1 范数的总和（若 filter_keys 指定则只统计匹配项）
    """
    total_l1 = 0.0
    if isinstance(params, dict):
        for key, val in params.items():
            new_prefix = prefix + key + '.'
            if isinstance(val, dict):
                sub_total = print_params_l1(val, new_prefix, filter_keys, total_only)
                total_l1 += sub_total
            elif isinstance(val, np.ndarray):
                # 如果指定了过滤器，检查键名是否包含任一子串
                if filter_keys is not None:
                    if not any(sub in key for sub in filter_keys):
                        continue
                l1 = np.sum(np.abs(val))
                total_l1 += l1
                if not total_only:
                    print(f"{new_prefix}{key} : {l1:.6f}")
            # 忽略其他类型（如 int, str, list 等）
    return total_l1
def save_feature_map_grid(
    tensor,
    save_path,
    d_mean=False,
    max_channels=16,
    percentile_clip=(1, 99),
    colormap='coolwarm',
    title=None,
    dpi=100
):
    """
    将特征图 (C, H, W) 或 (B, C, H, W) 保存为 PNG 图片文件。
    完全基于 NumPy + Matplotlib，无任何深度学习框架依赖。

    Args:
        tensor (np.ndarray): 特征图张量
        save_path (str): 保存路径，例如 './logs/features/step_100.png'
        max_channels (int): 最多显示通道数
        percentile_clip (tuple): 归一化裁剪百分位
        colormap (str): 颜色映射，如 'viridis', 'inferno', 'gray', 'RdYlBu'
        title (str): 可选标题，会显示在图片上方
        dpi (int): 图片分辨率
    """
    # ---------- 1. 形状处理（同之前的逻辑） ----------
    if tensor.ndim == 4:
        tensor = tensor[0]  # 取 batch 第一个
    elif tensor.ndim == 5:
        # 对于 5D，例如 (B, D, C, H, W)，自动对 D 维度取均值
        if d_mean:
            print(f"警告：输入为 5D 张量 {tensor.shape}，自动沿第1维 (D) 取均值")
            tensor = tensor.mean(axis=1)  # -> (B, C, H, W) 
            tensor = tensor[0]
        else:
            tensor = tensor[0][0]           # -> (C, H, W)
    elif tensor.ndim != 3:
        raise ValueError(f"输入必须是 3D 或 4D，当前为 {tensor.ndim}D")

    C, H, W = tensor.shape

    # ---------- 2. 通道采样 ----------
    if C > max_channels:
        indices = np.linspace(0, C-1, max_channels, dtype=int)
    else:
        indices = np.arange(C)
    n_channels = len(indices)
    selected = tensor[indices]

    # ---------- 3. 归一化与裁剪 ----------
    low, high = percentile_clip
    vmin = np.percentile(selected, low)
    vmax = np.percentile(selected, high)
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    selected = (selected - vmin) / (vmax - vmin)
    selected = np.clip(selected, 0.0, 1.0)

    # ---------- 4. 拼接网格 ----------
    cols = int(np.ceil(np.sqrt(n_channels)))
    rows = int(np.ceil(n_channels / cols))
    grid = np.zeros((rows * H, cols * W), dtype=np.float32)
    for i in range(n_channels):
        r = i // cols
        c = i % cols
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = selected[i]

    # ---------- 5. 绘图并保存 ----------
    fig, ax = plt.subplots(figsize=(cols * W / 100, rows * H / 100), dpi=dpi)
    ax.imshow(grid, cmap=colormap, aspect='auto')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)

    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)  # 关闭释放内存
def show_feature_map_live(
    tensor,
    max_channels=16,
    percentile_clip=(1, 99),
    colormap='viridis',
    title=None
):
    """
    实时显示特征图（会弹窗刷新）
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    elif tensor.ndim == 5:
        tensor = tensor.mean(axis=1)[0]
    C, H, W = tensor.shape

    if C > max_channels:
        indices = np.linspace(0, C-1, max_channels, dtype=int)
    else:
        indices = np.arange(C)
    selected = tensor[indices]

    low, high = percentile_clip
    vmin = np.percentile(selected, low)
    vmax = np.percentile(selected, high)
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    selected = (selected - vmin) / (vmax - vmin)
    selected = np.clip(selected, 0.0, 1.0)

    cols = int(np.ceil(np.sqrt(len(indices))))
    rows = int(np.ceil(len(indices) / cols))
    grid = np.zeros((rows * H, cols * W), dtype=np.float32)
    for i in range(len(indices)):
        r = i // cols
        c = i % cols
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = selected[i]

    plt.clf()  # 清空当前画布
    plt.imshow(grid, cmap=colormap, aspect='auto')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.pause(0.01)  # 短暂暂停，让 GUI 更新
def visualize_feature_map(
    tensor,
    max_channels=16,
    normalize=True,
    percentile_clip=(1, 99),
    colormap='viridis',
    return_type='rgb'
):
    """
    将卷积层特征图 (C, H, W) 或 (B, C, H, W) 转为网格 RGB 图像，用于 TensorBoard 或 Matplotlib 显示。

    Args:
        tensor (np.ndarray): 
            - 若为 (C, H, W)：直接使用
            - 若为 (B, C, H, W)：自动取 batch=0
            - 若为 (B, ..., C, H, W) 且无法自动推断，会报错提示
        max_channels (int): 最多显示多少个通道（若 C > max_channels，等间隔采样）
        normalize (bool): 是否归一化到 [0, 1]
        percentile_clip (tuple): (低百分位, 高百分位)，用于裁剪异常值，例如 (1, 99)
        colormap (str): matplotlib 颜色映射，如 'viridis', 'inferno', 'gray'
        return_type (str): 
            - 'rgb': 返回 (H_grid, W_grid, 3) 的 uint8 RGB 图
            - 'gray': 返回 (H_grid, W_grid) 的 float32 灰度图（未应用 colormap）

    Returns:
        np.ndarray: 网格图像（形状视 return_type 而定）
    """
    # ---------- 1. 形状检查和自动降维 ----------
    if tensor.ndim == 4:
        # (B, C, H, W) -> 取第一个样本
        tensor = tensor[0]
    elif tensor.ndim == 5:
        # 对于 (B, D, C, H, W) 这种，我们无法确定哪一维是通道。
        # 更合理的做法是让用户提前聚合（例如 mean 或取第一个 D），这里提示。
        raise ValueError(
            f"检测到 5D 张量 {tensor.shape}，无法自动判断通道轴。"
            "请先手动聚合，例如 tensor = tensor[0, 0] 或 tensor.mean(axis=1)"
        )
    elif tensor.ndim != 3:
        raise ValueError(f"输入必须是 (C, H, W) 或 (B, C, H, W)，当前为 {tensor.ndim}D")

    C, H, W = tensor.shape

    # ---------- 2. 通道采样 ----------
    if C > max_channels:
        # 等间隔采样，保留均匀分布
        indices = np.linspace(0, C-1, max_channels, dtype=int)
    else:
        indices = np.arange(C)
    n_channels = len(indices)
    selected = tensor[indices]  # (n_channels, H, W)

    # ---------- 3. 归一化与裁剪 ----------
    if normalize:
        low, high = percentile_clip
        vmin = np.percentile(selected, low)
        vmax = np.percentile(selected, high)
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-8
        # 裁剪并归一化到 [0, 1]
        selected = (selected - vmin) / (vmax - vmin)
        selected = np.clip(selected, 0.0, 1.0)
    else:
        # 若不做归一化，有些值可能超出 [0,1]，直接 clip 到 [0,1] 以防显示异常
        selected = np.clip(selected, 0.0, 1.0)

    # ---------- 4. 拼成网格 ----------
    cols = int(np.ceil(np.sqrt(n_channels)))
    rows = int(np.ceil(n_channels / cols))
    grid = np.zeros((rows * H, cols * W), dtype=np.float32)

    for i in range(n_channels):
        r = i // cols
        c = i % cols
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = selected[i]

    # ---------- 5. 返回不同格式 ----------
    if return_type == 'gray':
        return grid  # (H_grid, W_grid) float32

    # 默认返回 RGB：应用颜色映射
    cmap_obj = plt.cm.get_cmap(colormap)
    grid_rgb = cmap_obj(grid)[:, :, :3]  # 去掉 alpha 通道
    grid_rgb = (grid_rgb * 255).astype(np.uint8)
    return grid_rgb
