import numpy as np
import math
from model import init_model_params, bev_forward, bev_backward, compute_losses, d_compute_losses

def recursive_sgd_update_(params, grads, lr):
    """递归更新嵌套参数字典"""
    for key in params:
        if key in grads:
            if isinstance(params[key], dict):
                recursive_sgd_update(params[key], grads[key], lr)
            else:
                params[key] -= lr * grads[key]
def recursive_sgd_update(params, grads, lr):
    """
    递归更新嵌套参数字典，支持列表类型。
    """
    for key in params:
        if key not in grads:
            continue
        if isinstance(params[key], dict):
            recursive_sgd_update(params[key], grads[key], lr)
        elif isinstance(params[key], list):
            # 列表类型：逐个元素更新
            if isinstance(grads[key], list):
                for i in range(len(params[key])):
                    if i < len(grads[key]):
                        params[key][i] -= lr * grads[key][i]
            else:
                # 如果梯度不是列表但参数是列表（罕见），直接赋值
                params[key] = [p - lr * g for p, g in zip(params[key], grads[key])]
        else:
            params[key] -= lr * grads[key]
def adam_update(params, grads, lr, step, m=None, v=None, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    递归更新嵌套参数字典，支持列表。
    params: 嵌套参数字典（可包含列表）
    grads: 与 params 结构相同的梯度
    step: 当前步数（从1开始）
    m, v: 动量缓存（与 params 结构相同），若为 None 则初始化
    返回: (params, m, v, step+1)
    """
    if m is None:
        m = {}
    if v is None:
        v = {}

    def _update(p, g, m_cur, v_cur, lr_scale):
        if isinstance(p, dict):
            # 确保 m_cur 和 v_cur 是字典
            if not isinstance(m_cur, dict):
                m_cur = {}
            if not isinstance(v_cur, dict):
                v_cur = {}
            for key in p:
                if key in g:
                    m_cur[key], v_cur[key] = _update(
                        p[key], g[key],
                        m_cur.get(key, {}),
                        v_cur.get(key, {}),
                        lr_scale
                    )
        elif isinstance(p, list):
            if not isinstance(g, list) or len(p) != len(g):
                raise ValueError("参数和梯度列表长度不匹配")
            # 确保 m_cur 和 v_cur 是列表
            if not isinstance(m_cur, list):
                m_cur = [None] * len(p)
            if not isinstance(v_cur, list):
                v_cur = [None] * len(p)
            for i in range(len(p)):
                m_cur[i], v_cur[i] = _update(
                    p[i], g[i],
                    m_cur[i] if i < len(m_cur) else None,
                    v_cur[i] if i < len(v_cur) else None,
                    lr_scale
                )
        else:
            # 数值更新
            # 确保 m_cur 和 v_cur 是数组
            if not isinstance(m_cur, np.ndarray):
                m_cur = np.zeros_like(p)
            if not isinstance(v_cur, np.ndarray):
                v_cur = np.zeros_like(p)
            m_cur = beta1 * m_cur + (1 - beta1) * g
            v_cur = beta2 * v_cur + (1 - beta2) * (g * g)
            m_hat = m_cur / (1 - beta1 ** step)
            v_hat = v_cur / (1 - beta2 ** step)
            p -= lr * lr_scale * m_hat / (np.sqrt(v_hat) + eps)
        return m_cur, v_cur

    _update(params, grads, m, v, 1.0)
    return params, m, v, step + 1
def clip_gradients(grads, max_norm=1.0):
    """对嵌套字典中的梯度进行裁剪（按全局范数）"""
    # 计算总范数
    total_norm = 0.0
    def accumulate_norm(g):
        nonlocal total_norm
        if isinstance(g, dict):
            for v in g.values():
                accumulate_norm(v)
        elif isinstance(g, list):
            for v in g:
                accumulate_norm(v)
        else:
            total_norm += np.sum(g ** 2)
    accumulate_norm(grads)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        def scale_grads(g):
            if isinstance(g, dict):
                for k in g:
                    scale_grads(g[k])
            elif isinstance(g, list):
                for i in range(len(g)):
                    scale_grads(g[i])
            else:
                g *= scale
        scale_grads(grads)
    return grads
def cosine_annealing(epoch, total_epochs, lr_init=1e-3, lr_min=1e-6):
    """余弦退火学习率"""
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(math.pi * epoch / total_epochs))

def main():
    bev_shape = (200, 200)
#    B, N, H, W = 2, 6, 256, 704
    B, N, H, W = 2, 6, 64, 176
    depth_bins = 41
    context_channels = 64
    bev_channels = 64
    num_classes = 10

    # 初始化参数
    model_params = init_model_params(
        backbone_in=3,
        backbone_out=64,
        depth_bins=depth_bins,
        context_channels=context_channels,
        bev_channels=bev_channels,
        bev_shape=bev_shape,
        num_classes=num_classes
    )

    # 固定几何索引（模拟）
    Hf, Wf = H // 32, W // 32
    N_points = depth_bins * Hf * Wf
    geom_indices_list = []
    for b in range(B):
        cam_indices = []
        for n in range(N):
            idx = np.random.randint(0, bev_shape[0]*bev_shape[1], size=N_points)
            cam_indices.append(idx)
        geom_indices_list.append(cam_indices)

    # 固定输入图像（随机但固定）
    np.random.seed(42)
    images = np.random.randn(B, N, 3, H, W).astype(np.float32)
    img_min = images.min()
    img_max = images.max()
    images = 2*(images - img_min) / (img_max-img_min) -1


    # 固定 ground truth：在 BEV 中心放置一个物体
    def gaussian_2d(shape, center, sigma=2.0):
        y, x = np.ogrid[:shape[0], :shape[1]]
        x0, y0 = center
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    heatmap_gt = np.zeros((B, num_classes, bev_shape[0], bev_shape[1]), dtype=np.float32)
    center_x, center_y = bev_shape[0]//2, bev_shape[1]//2
    heatmap_gt[b,0,center_x,center_y]=1.0
#    for b in range(B):
#        for ni in range(num_classes):
#            heatmap_gt[b, ni] = gaussian_2d(bev_shape, (center_x, center_y), sigma=20.0)
#            heatmap_gt[b, ni] = (heatmap_gt[b, ni]>0.5).astype(np.float32)
	


    reg_gt = np.zeros((B, 8, bev_shape[0], bev_shape[1]), dtype=np.float32)
    #depth_gt = np.random.randint(0, depth_bins, size=(B, Hf, Wf)).astype(np.int32)
    # 构造一个人工的深度图（中心区域深度为20，其余为忽略）
    depth_gt = np.full((B, Hf, Wf), -1, dtype=np.int32)  # -1 表示忽略
    center_h, center_w = Hf//2, Wf//2
    radius = min(center_h, center_w, Hf-center_h-1, Wf-center_w-1, 2)  # 最多2，且不越界
    for dh in range(-radius,radius+1):
        for dw in range(-radius,radius+1):
            depth_gt[:, center_h+dh, center_w+dw] = 20  # 固定深度值
    reg_gt[:, 0, center_x, center_y] = 0.0   # dx
    reg_gt[:, 1, center_x, center_y] = 0.0   # dy
    reg_gt[:, 2, center_x, center_y] = 2.0   # w
    reg_gt[:, 3, center_x, center_y] = 4.0   # l
    reg_gt[:, 4, center_x, center_y] = 1.5   # h
    reg_gt[:, 5, center_x, center_y] = 0.0   # sinθ
    reg_gt[:, 6, center_x, center_y] = 1.0   # cosθ
    reg_gt[:, 7, center_x, center_y] = 1.0   # depth (物体中心高度)
    # 训练循环
    lr_init = 5e-2
    epochs = 200
    m,v = None,None
    step = 1

    for epoch in range(epochs):
        lr = cosine_annealing(epoch,epochs,lr_init=lr_init,lr_min=1e-6)
        # 前向
        heatmap, reg, depth_logits_list, bev_feat, caches = bev_forward(
            images, geom_indices_list, bev_shape, model_params
        )
        # 堆叠深度 logits（取每个 batch 的第一个相机）
        depth_logits_batch = np.stack([depth_logits_list[b * N] for b in range(B)], axis=0)

        # 计算损失
        losses, loss_caches = compute_losses(heatmap, reg, heatmap_gt, reg_gt, depth_logits_batch, depth_gt)

        # 计算梯度
        dheatmap, dreg, ddepth = d_compute_losses(
            heatmap, reg, heatmap_gt, reg_gt, depth_logits_batch, depth_gt, loss_caches
        )
        # 拆分 ddepth 为每个相机的梯度 (每个形状 (1, D, Hf, Wf))
        ddepth_per_cam = []
        for b in range(B):
            for n in range(N):
                if n==0:
                    ddepth_per_cam.append(ddepth[b][None, ...])  # ddepth[b] 是 (D, H, W)
                else:
                    ddepth_per_cam.append(None)

        # 反向传播
        grads = bev_backward(
            dheatmap, dreg, ddepth_per_cam,
            caches, model_params, geom_indices_list, bev_shape
        )
        # 梯度范数
        total_grad_norm = 0.0
        def accumulate_norm(g):
            nonlocal total_grad_norm
            if isinstance(g, dict):
                for v in g.values():
                    accumulate_norm(v)
            elif isinstance(g, list):
                for v in g:
                    accumulate_norm(v)
            else:
                total_grad_norm += np.sum(g ** 2)
        accumulate_norm(grads)
        total_grad_norm = np.sqrt(total_grad_norm)
        grads = clip_gradients(grads, max_norm=10.0)
        # 更新参数（SGD）
#        recursive_sgd_update(model_params, grads, lr)
        model_params,m,v,step = adam_update(model_params,grads,lr,step,m,v)
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch:3d}, total: {losses['total_loss']:.6f},hm:{losses['loss_heatmap']:.6f},reg:{losses['loss_reg']:.6f},depth:{losses['loss_depth']:.6f},grad_norm: {total_grad_norm:.6f}")

    print("训练完成！")

if __name__ == "__main__":
    main()
