import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from funcs import (
    conv, d_conv, silu, d_silu, avgpool, d_avgpool,
    softmax, d_softmax, sigmoid, linear, d_linear
)

# ============================
# 1. 图像特征提取（ResNet18 简化版）
# ============================

def resnet_stem(x, params):
    """
    x: (B, 3, H, W)
    params: {'conv1_w': (64,3,7,7), 'conv1_b': (64,)}
    返回: out (B,64,H/4,W/4), cache (x, out_conv, out_relu, out_pool)
    """
    out_conv, _, _ = conv(x, params['conv1_w'], params['conv1_b'], stride=2, padding=3)
    out_relu = silu(out_conv)
    out_pool, avg_cache = avgpool(out_relu, kernel_size=3, stride=2, padding=1)
    cache = (x, out_conv, out_relu, out_pool,avg_cache)
    return out_pool, cache

def d_resnet_stem(dout, cache, params):
    x, out_conv, out_relu, out_pool,avg_cache = cache
    # avgpool 反向
    d_pool = d_avgpool(dout, avg_cache)
    # relu 反向
    d_relu_out = d_silu(d_pool, out_relu)
    # conv 反向
    dx, dw, db = d_conv(d_relu_out, x, params['conv1_w'], stride=2, padding=3)
    grads = {'conv1_w': dw, 'conv1_b': db}
    return dx, grads

def resnet_block(x, params, block_id):
    """
    x: (B, C, H, W)
    params: {'conv1_w': (C,C,3,3), 'conv1_b': (C,), 'conv2_w': (C,C,3,3), 'conv2_b': (C,)}
    返回: out (B,C,H,W), cache (x, out1, out2, out_final)
    """
    out1_conv, _, _ = conv(x, params['conv1_w'], params['conv1_b'], stride=1, padding=1)
    out1 = silu(out1_conv)
    out2_conv, _, _ = conv(out1, params['conv2_w'], params['conv2_b'], stride=1, padding=1)
    out2 = out2_conv + x
    out_final = silu(out2)
    cache = (x, out1, out2, out_final)
    return out_final, cache

def d_resnet_block(dout, cache, params, block_id):
    x, out1, out2, out_final = cache
    # 残差后 ReLU 反向
    d_relu_final = d_silu(dout, out_final)
    # 残差连接分配
    d_conv2_out = d_relu_final
    d_residual = d_relu_final
    # conv2 反向
    dx_conv2, dw2, db2 = d_conv(d_conv2_out, out1, params['conv2_w'], stride=1, padding=1)
    # conv1 反向
    d_relu1 = d_silu(dx_conv2, out1)
    dx_conv1, dw1, db1 = d_conv(d_relu1, x, params['conv1_w'], stride=1, padding=1)
    dx = dx_conv1 + d_residual
    grads = {
        'conv1_w': dw1, 'conv1_b': db1,
        'conv2_w': dw2, 'conv2_b': db2
    }
    return dx, grads

def resnet18_features(x, params):
    """
    x: (B, 3, H, W)
    params: 包含 'stem' 和 'stage0'...'stage3'，每个阶段包含两个残差块参数
    返回: features (B, 512, H//32, W//32), caches 字典
    """
    caches = {}
    out, stem_cache = resnet_stem(x, params['stem'])
    caches['stem'] = stem_cache
    avg_caches = []
    for stage in range(4):
        stage_caches = []
        for block in range(2):
            block_params = params[f'stage{stage}']
            # 注意：每个块使用相同的参数名（conv1_w, conv1_b, conv2_w, conv2_b）
            out, block_cache = resnet_block(out, block_params, block)
            stage_caches.append(block_cache)
        caches[f'stage{stage}'] = stage_caches
        if stage < 3:
            out, avg_cache = avgpool(out, kernel_size=2, stride=2)
            avg_caches.append(avg_cache)
    caches['avg_caches']=avg_caches
    return out, caches

def d_resnet18_features(dout, caches, params):
    """
    dout: (B, 512, Hf, Wf)
    caches: 前向保存的缓存字典
    params: backbone 参数字典
    返回: dx (对输入图像的梯度), grads 字典
    """
    dx = dout
    grads = {}
    avg_caches = caches['avg_caches']
    # 从后往前遍历阶段
    for stage in range(3, -1, -1):
        # 如果该 stage 有下采样（stage < 3），前向有 avgpool，反向需先上采样
        if stage < 3:
            avg_cache = avg_caches[stage]
            dx = d_avgpool(dx,avg_cache)
        stage_caches = caches[f'stage{stage}']  # list
        # 块反向
        for block_idx in range(len(stage_caches)-1, -1, -1):
            block_cache = stage_caches[block_idx]
            block_params = params[f'stage{stage}']
            dx, block_grads = d_resnet_block(dx, block_cache, block_params, block_idx)
            # 合并梯度
            for k, v in block_grads.items():
                if k not in grads:
                    grads[k] = v
                else:
                    grads[k] += v
    # stem 反向
    dx_stem, stem_grads = d_resnet_stem(dx, caches['stem'], params['stem'])
    grads.update(stem_grads)
    return dx_stem, grads

# ============================
# 2. Lift 模块
# ============================

def lift(features, depth_params, context_params):
    """
    features: (B, C, H, W)
    depth_params: {'w': (D, C, 3,3), 'b': (D,)}
    context_params: {'w': (C_ctx, C, 3,3), 'b': (C_ctx,)}
    返回: lift_feat (B, D, C_ctx, H, W), depth_logits (B, D, H, W), cache
    """
    depth_logits, _, _ = conv(features, depth_params['w'], depth_params['b'], stride=1, padding=1)
    depth_prob, _ = softmax(depth_logits, axis=1)
    context, _, _ = conv(features, context_params['w'], context_params['b'], stride=1, padding=1)
    B, D, H, W = depth_prob.shape
    C_ctx = context.shape[1]
    lift_feat = depth_prob[:, :, None, :, :] * context[:, None, :, :, :]
    cache = (depth_logits, depth_prob, context, depth_params, context_params)
    return lift_feat, depth_logits, cache

def d_lift(dout_lift_feat, depth_logits, depth_prob, context, features, depth_params, context_params):
    """
    完整 Lift 反向
    dout_lift_feat: (B, D, C_ctx, H, W)
    返回: d_features, grads_depth, grads_context
    """
    B, D, C_ctx, H, W = dout_lift_feat.shape
    # 1. 外积反向
    ddepth_prob = np.sum(dout_lift_feat * context[:, None, :, :, :], axis=2)  # (B,D,H,W)
    dcontext = np.sum(dout_lift_feat * depth_prob[:, :, None, :, :], axis=1)  # (B,C_ctx,H,W)
    # 2. depth_logits 梯度
    d_depth_logits = d_softmax(ddepth_prob, depth_prob)
    # 3. 深度卷积反向
    dx_depth, dw_depth, db_depth = d_conv(d_depth_logits, features, depth_params['w'], stride=1, padding=1)
    # 4. 上下文卷积反向
    dx_context, dw_context, db_context = d_conv(dcontext, features, context_params['w'], stride=1, padding=1)
    d_features = dx_depth + dx_context
    grads_depth = {'w': dw_depth, 'b': db_depth}
    grads_context = {'w': dw_context, 'b': db_context}
    return d_features, grads_depth, grads_context

# ============================
# 3. BEV Pooling
# ============================

def bev_pool(lift_feat, geom_indices, bev_shape):
    """
    lift_feat: (B, D, C, H, W)
    geom_indices: (N,) 一维网格索引
    bev_shape: (X, Y)
    """
    B, D, C_ctx, H, W = lift_feat.shape
    N = D * H * W
    lift_flat = lift_feat.reshape(B, N, C_ctx)
    X, Y = bev_shape
    bev_list = []
    for b in range(B):
        feat = lift_flat[b]
        # 按网格索引排序
        order = np.argsort(geom_indices)
        sorted_feat = feat[order]
        sorted_idx = geom_indices[order]
        # 获取唯一网格和每个网格的点数
        unique_idx, counts = np.unique(sorted_idx, return_counts=True)
        # 计算累积和（cumsum trick）
        cumsum = np.cumsum(sorted_feat, axis=0)  # (N, C)
        # 每个网格的结束索引（0-based）
        ends = np.cumsum(counts) - 1  # 每个网格最后一个点的索引
        # 计算每个网格的起始索引
        starts = np.concatenate(([0], ends[:-1] + 1))  # 每个网格第一个点的索引
        # 利用 diff 提取每个网格的和：
        # 方法：在 ends 位置取 cumsum 的差值
        # 构造一个包含 ends 和 starts 的索引数组，用 diff 实现区间和
        # 更标准：grid_sums = cumsum[ends] - cumsum[starts-1]（处理 starts=0 时视为 -1）
        # 但为了完全符合 cumsum trick，我们可以使用 np.diff 在 ends 处取差分
        # 由于 cumsum 是累积和，网格和 = cumsum[ends] - cumsum[starts-1]
        # 我们可以用 np.diff 在 ends 处取差分，但需要预先补0
        # 这里我们用 np.diff 对 cumsum 在 ends 处取差分，并乘以计数
        # 实际上，更简洁的 cumsum diff 方式：
        # 构造一个起始和结束的索引数组，用 np.diff 实现区间和
        # 通常的做法：grid_sums = cumsum[ends] - np.concatenate(([0], cumsum[ends[:-1]]))
        # 这样等价于 cumsum[ends] - cumsum[starts-1]
        grid_sums = cumsum[ends] - np.concatenate((np.zeros((1,C_ctx)), cumsum[ends[:-1]]))
        # 验证：对于每个网格，其和为 cumsum[end] - cumsum[start-1]，这里 start-1 = 前一个网格的 end
        # 所以 grid_sums[i] = cumsum[ends[i]] - cumsum[ends[i-1]]（当 i>0 时）
        # 这与 cumsum diff 完全一致。
        bev = np.zeros((X * Y, C_ctx), dtype=np.float32)
        bev[unique_idx] = grid_sums
        bev = bev.reshape(X, Y, C_ctx).transpose(2, 0, 1)
        bev_list.append(bev)
    return np.stack(bev_list, axis=0)

def d_bev_pool(dout_bev, lift_feat, geom_indices, bev_shape):
    """
    dout_bev: (B, C_ctx, X, Y)
    lift_feat: (B, D, C_ctx, H, W)
    geom_indices: (N,)
    bev_shape: (X, Y)
    返回: (B, D, C_ctx, H, W)
    """
    B, C_ctx, X, Y = dout_bev.shape
    B, D, C_ctx, H, W = lift_feat.shape
    N = D * H * W
    dout_flat = dout_bev.reshape(B, C_ctx, -1).transpose(0, 2, 1)  # (B, X*Y, C)
    d_lift_flat = np.zeros((B, N, C_ctx), dtype=np.float32)
    for b in range(B):
        for grid_idx in range(X * Y):
            mask = geom_indices == grid_idx
            if np.any(mask):
                d_lift_flat[b, mask] = dout_flat[b, grid_idx]
    d_lift = d_lift_flat.reshape(B, D, H, W, C_ctx).transpose(0, 1, 4, 2, 3)
    return d_lift

# ============================
# 4. BEV 编码器
# ============================

def bev_encoder(bev_feat, params):
    """
    bev_feat: (B, C_in, X, Y)
    params: {'weights': list, 'biases': list}
    返回: (B, C_out, X, Y), caches 列表
    """
    out = bev_feat
    caches = []
    for w, b in zip(params['weights'], params['biases']):
        out_conv, _, _ = conv(out, w, b, stride=1, padding=1)
        out_relu = silu(out_conv)
        caches.append((out, out_conv, out_relu))
        out = out_relu
    return out, caches

def d_bev_encoder(dout, caches, params):
    """
    dout: (B, C_out, X, Y)
    caches: 列表，每项 (input, conv_out, relu_out)
    params: {'weights': list, 'biases': list}
    返回: dbev_in, grads (字典包含 'weights', 'biases')
    """
    dx = dout
    grads_w = []
    grads_b = []
    for i in range(len(caches)-1, -1, -1):
        inp, conv_out, relu_out = caches[i]
        w = params['weights'][i]
        b = params['biases'][i]
        d_x_relu = d_silu(dx, relu_out)
        d_conv_in, dw, db = d_conv(d_x_relu, inp, w, stride=1, padding=1)
        dx = d_conv_in
        grads_w.append(dw)
        grads_b.append(db)
    grads_w.reverse()
    grads_b.reverse()
    grads = {'weights': grads_w, 'biases': grads_b}
    return dx, grads

# ============================
# 5. CenterNet 检测头
# ============================

def centernet_head(bev_feat, head_params):
    """
    bev_feat: (B, C, X, Y)
    head_params: {'heatmap_w': (num_classes, C, 3,3), 'heatmap_b': (num_classes,),
                 'reg_w': (8, C, 3,3), 'reg_b': (8,)}
    返回: heatmap (B, num_classes, X, Y), reg (B, 8, X, Y)
    """
    heatmap, _, _ = conv(bev_feat, head_params['heatmap_w'], head_params['heatmap_b'], stride=1, padding=1)
    reg, _, _ = conv(bev_feat, head_params['reg_w'], head_params['reg_b'], stride=1, padding=1)
    return heatmap, reg

def d_centernet_head(dheatmap, dreg, bev_feat, head_params):
    """
    dheatmap: (B, num_classes, X, Y)
    dreg: (B, 8, X, Y)
    bev_feat: (B, C, X, Y)
    head_params: 参数字典
    返回: dbev, grads 字典
    """
    dbev_hm, dw_hm, db_hm = d_conv(dheatmap, bev_feat, head_params['heatmap_w'], stride=1, padding=1)
    dbev_reg, dw_reg, db_reg = d_conv(dreg, bev_feat, head_params['reg_w'], stride=1, padding=1)
    dbev = dbev_hm + dbev_reg
    grads = {
        'heatmap_w': dw_hm, 'heatmap_b': db_hm,
        'reg_w': dw_reg, 'reg_b': db_reg
    }
    return dbev, grads

# ============================
# 6. 损失函数及其梯度
# ============================
def focal_loss(heatmap_pred, heatmap_gt, alpha=0.25, gamma=2.0):
    """
    标准 Focal Loss（二分类）
    heatmap_gt: (B, C, H, W) 二值标签（0或1）
    """
    pred = sigmoid(heatmap_pred)
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    pos_mask = heatmap_gt == 1
    neg_mask = heatmap_gt == 0
    # 计算 p_t
    p_t = pred * pos_mask + (1 - pred) * neg_mask
    # 平衡权重 alpha_t
    alpha_t = alpha * pos_mask + (1 - alpha) * neg_mask
    # focal loss
    loss = -alpha_t * (1 - p_t) ** gamma * np.log(p_t + 1e-8)
    return np.mean(loss)

def d_focal_loss(heatmap_pred, heatmap_gt, alpha=0.25, gamma=2.0):
    pred = sigmoid(heatmap_pred)
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    pos_mask = (heatmap_gt == 1).astype(np.float32)
    neg_mask = (heatmap_gt == 0).astype(np.float32)
    p_t = pred * pos_mask + (1 - pred) * neg_mask
    alpha_t = alpha * pos_mask + (1 - alpha) * neg_mask
    # 梯度公式：dL/dx = (p_t - y) * alpha_t * (1 - p_t) ** gamma * (gamma * p_t * log(p_t) + p_t - 1)  # 需要推导
    # 这里采用数值稳定的方式计算梯度
    # 为了避免复杂的符号，我们直接使用数值梯度（但效率低），或使用自动微分模拟
    # 由于我们需要梯度，直接计算：
    grad = -alpha_t * (1 - p_t) ** gamma * (1 - gamma * p_t * (np.log(p_t + 1e-8) + 1) - p_t * np.log(p_t + 1e-8))
    # 简化梯度（标准实现通常用以下形式）：
    # grad = -alpha_t * (1 - p_t) ** gamma * (gamma * p_t * np.log(p_t) + p_t - 1)
    # 但更稳健的做法是使用有限差分，但为了效率，我们采用常见的梯度公式：
    # 来自 https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    grad = -alpha_t * (1 - p_t) ** gamma * (gamma * p_t * np.log(p_t + 1e-8) + p_t - 1) * (2 * pos_mask - 1)
    # 裁剪梯度
    grad = np.clip(grad, -10.0, 10.0)
    grad = grad - grad
    return grad
def focal_loss_(heatmap_pred, heatmap_gt, alpha=2.0, beta=4.0):
    pred = sigmoid(heatmap_pred)
#    pred = np.clip(pred,1e-7,1-1e-7)
    pos_mask = heatmap_gt == 1
    neg_mask = heatmap_gt == 0
    loss_pos = - (1 - pred) ** alpha * np.log(pred + 1e-6) * pos_mask
    loss_neg = - pred ** alpha * np.log(1 - pred + 1e-6) * neg_mask * (1 - heatmap_gt) ** beta
    return np.mean(loss_pos + loss_neg)
def d_focal_loss_(heatmap_pred, heatmap_gt, alpha=2.0, beta=4.0):
    pred = sigmoid(heatmap_pred)
    pred = np.clip(pred,1e-7,1-1e-7)
    pos_mask = (heatmap_gt == 1).astype(np.float32)
    neg_mask = (heatmap_gt == 0).astype(np.float32)
    # 计算每个位置的梯度（向量化）
    grad_pos = - (1 - pred) ** alpha * (alpha * pred * np.log(pred + 1e-6) - pred + 1)
    grad_neg = pred ** alpha * (alpha * (1 - pred) * np.log(1 - pred + 1e-6) - pred) * (1 - heatmap_gt) ** beta
    grad = pos_mask * grad_pos + neg_mask * grad_neg
    return grad
# 修改 reg_l1_loss 和 d_reg_l1_loss
def reg_l1_loss(reg_pred, reg_gt, heatmap_gt):
    # heatmap_gt: (B, num_classes, X, Y)
    pos_mask = np.sum(heatmap_gt, axis=1) > 0  # (B, X, Y)
    if np.sum(pos_mask) == 0:
        return 0.0
    # 扩展掩码到 (B, 8, X, Y)
    print(pos_mask.sum())
    pos_mask_expanded = np.repeat(pos_mask[:, None, :, :], reg_pred.shape[1], axis=1)  # (B, 8, X, Y)
    print(reg_pred[pos_mask_expanded] - reg_gt[pos_mask_expanded])
    return np.mean(np.abs(reg_pred[pos_mask_expanded] - reg_gt[pos_mask_expanded]))

def d_reg_l1_loss(reg_pred, reg_gt, heatmap_gt):
    pos_mask = np.sum(heatmap_gt, axis=1) > 0
    grad = np.zeros_like(reg_pred)
    if np.sum(pos_mask) == 0:
        return grad
    pos_mask_expanded = np.repeat(pos_mask[:, None, :, :], reg_pred.shape[1], axis=1)
    grad[pos_mask_expanded] = np.sign(reg_pred[pos_mask_expanded] - reg_gt[pos_mask_expanded]) / np.sum(pos_mask)
    return grad

def depth_loss(depth_pred, depth_gt):
    """
    depth_pred: (B, D, H, W) logits
    depth_gt: (B, H, W) 整数索引，-1 表示忽略
    """
    B, D, H, W = depth_pred.shape
    pred_flat = depth_pred.transpose(0, 2, 3, 1).reshape(-1, D)
    gt_flat = depth_gt.reshape(-1)
    valid = gt_flat >= 0
    if np.sum(valid) == 0:
        return 0.0, None
    probs, _ = softmax(pred_flat, axis=1)
    gt_valid = gt_flat[valid]
    log_probs = np.log(probs[valid] + 1e-8)
    loss = -np.mean(log_probs[np.arange(len(gt_valid)), gt_valid])
    cache = (valid, probs, gt_valid, B, D)   # 只缓存 5 个
    return loss, cache
def d_depth_loss(depth_pred, depth_gt, cache=None):
    """
    depth_pred: (B, D, H, W) logits
    depth_gt: (B, H, W) 标签
    cache: 从 depth_loss 返回的缓存，若提供则复用
    返回: (B, D, H, W) 梯度
    """
    B, D, H, W = depth_pred.shape
    if cache is not None:
        valid, probs, gt_valid, B, D = cache
    else:
        pred_flat = depth_pred.transpose(0, 2, 3, 1).reshape(-1, D)
        gt_flat = depth_gt.reshape(-1)
        valid = gt_flat >= 0
        if np.sum(valid) == 0:
            return np.zeros_like(depth_pred)
        probs, _ = softmax(pred_flat, axis=1)
        gt_valid = gt_flat[valid]

    N_valid = np.sum(valid)
    if N_valid == 0:
        return np.zeros_like(depth_pred)

    grad_flat = np.zeros((B*H*W, D), dtype=np.float32)
    one_hot = np.zeros((N_valid, D), dtype=np.float32)
    one_hot[np.arange(N_valid), gt_valid] = 1.0
    grad_flat[valid] = (probs[valid] - one_hot) / N_valid
    grad = grad_flat.reshape(B, H, W, D).transpose(0, 3, 1, 2)
    grad = grad - grad
    return grad
def compute_losses(heatmap_pred, reg_pred, heatmap_gt, reg_gt, depth_pred=None, depth_gt=None,
                   hm_weight=1.0, reg_weight=1.0, depth_weight=1.0):
    loss_hm = hm_weight * focal_loss(heatmap_pred, heatmap_gt)
    loss_reg = reg_weight * reg_l1_loss(reg_pred, reg_gt, heatmap_gt)
    depth_cache = None
    if depth_pred is not None and depth_gt is not None:
        loss_depth, depth_cache = depth_loss(depth_pred, depth_gt)
        loss_depth = depth_weight * loss_depth
    else:
        loss_depth = 0.0
    total = loss_hm + loss_reg + loss_depth
    caches = {'depth': depth_cache}
    return {'loss_heatmap': loss_hm, 'loss_reg': loss_reg, 'loss_depth': loss_depth, 'total_loss': total}, caches
def d_compute_losses(heatmap_pred, reg_pred, heatmap_gt, reg_gt,
                     depth_pred=None, depth_gt=None, caches=None,
                     hm_weight=1.0, reg_weight=1.0, depth_weight=1.0):
    # 梯度需要乘以对应的权重
    dheatmap = d_focal_loss(heatmap_pred, heatmap_gt) * hm_weight
    dreg = d_reg_l1_loss(reg_pred, reg_gt, heatmap_gt) * reg_weight
    ddepth = None
    if depth_pred is not None and depth_gt is not None:
        depth_cache = caches.get('depth') if caches else None
        ddepth = d_depth_loss(depth_pred, depth_gt, depth_cache)
        ddepth *= depth_weight
    return dheatmap, dreg, ddepth
# ============================
# 7. 完整 BEV 模型
# ============================

def bev_forward(images, geom_indices_list, bev_shape, model_params):
    """
    images: (B, 6, 3, H, W)
    geom_indices_list: list of list of arrays, shape (B, 6) 每个相机的网格索引
    bev_shape: (X, Y)
    model_params: 包含所有参数
    返回: heatmap, reg, depth_logits_list, bev_feat, caches
    """
    B, N, C, H, W = images.shape
    caches = {}
    # 1. 特征提取
    feat_list = []
    backbone_caches = []
    for b in range(B):
        for n in range(N):
            img = images[b, n]
            feat, cache = resnet18_features(img[None, ...], model_params['backbone'])
            feat_list.append(feat)
            backbone_caches.append(cache)
    features = np.stack(feat_list, axis=0).reshape(B, N, feat.shape[1], feat.shape[2], feat.shape[3])
    caches['backbone_caches'] = backbone_caches
    caches['features'] = features

    # 2. Lift
    lift_feats = []
    depth_logits_list = []
    lift_caches = []
    for b in range(B):
        for n in range(N):
            feat = features[b, n][None,...]
            lift_feat, depth_logits, lift_cache = lift(feat, model_params['lift_depth'], model_params['lift_context'])
            lift_feats.append(lift_feat[0])
            depth_logits_list.append(depth_logits[0])
            lift_caches.append(lift_cache)
    lift_feats = np.stack(lift_feats, axis=0)  # (B*N, D, C_ctx, Hf, Wf)
    caches['lift_caches'] = lift_caches
    caches['lift_feats'] = lift_feats
    # 3. BEV Pooling
    bev_acc = np.zeros((B, lift_feats.shape[2], bev_shape[0], bev_shape[1]), dtype=np.float32)
    for b in range(B):
        for n in range(N):
            idx = b * N + n
            geom_idx = geom_indices_list[b][n]
            lift_feat = lift_feats[idx]
            bev_cam = bev_pool(lift_feat[None,...], geom_idx, bev_shape)
            bev_acc[b] += bev_cam[0]
    caches['bev_acc'] = bev_acc

    # 4. BEV 编码
    bev_feat, bev_encoder_caches = bev_encoder(bev_acc, model_params['bev_encoder'])
    caches['bev_encoder_caches'] = bev_encoder_caches
    caches['bev_feat'] = bev_feat

    # 5. 检测头
    heatmap, reg = centernet_head(bev_feat, model_params['head'])
    caches['heatmap'] = heatmap
    caches['reg'] = reg

    return heatmap, reg, depth_logits_list, bev_feat, caches

def bev_backward(dheatmap, dreg, ddepth_list, caches, model_params, geom_indices_list, bev_shape):
    """
    dheatmap: (B, num_classes, X, Y)
    dreg: (B, 8, X, Y)
    ddepth_list: list of (B, D, Hf, Wf) 或 None (每个相机)
    caches: bev_forward 返回的缓存
    model_params: 参数字典
    geom_indices_list: 同前向
    bev_shape: (X, Y)
    返回: grads 字典
    """
    B, N, _, _, _ = caches['features'].shape
    grads = {}

    # 1. 检测头反向
    bev_feat = caches['bev_feat']
    head_params = model_params['head']
    dbev, head_grads = d_centernet_head(dheatmap, dreg, bev_feat, head_params)
    grads['head'] = head_grads

    # 2. BEV 编码器反向
    bev_encoder_caches = caches['bev_encoder_caches']
    bev_enc_params = model_params['bev_encoder']
    dbev_enc_in, bev_enc_grads = d_bev_encoder(dbev, bev_encoder_caches, bev_enc_params)
    grads['bev_encoder'] = bev_enc_grads

    # 3. BEV Pooling 反向
    # 将 dbev_enc_in (B,C,X,Y) 复制到每个相机
    dbev_per_cam = np.repeat(dbev_enc_in[:, None, :, :, :], N, axis=1)  # (B,N,C,X,Y)
    dbev_flat = dbev_per_cam.reshape(-1, dbev_enc_in.shape[1], dbev_enc_in.shape[2], dbev_enc_in.shape[3])  # (B*N,C,X,Y)
    lift_feats = caches['lift_feats']  # (B*N, D, C_ctx, Hf, Wf)
    dlift_feats = np.zeros_like(lift_feats)
    for idx in range(B*N):
        b = idx // N
        n = idx % N
        geom_idx = geom_indices_list[b][n]
        dbev_cam = dbev_flat[idx][None, ...]  # (1,C,X,Y)
        lift_feat_cam = lift_feats[idx][None, ...]  # (1,D,C_ctx,H,W)
        dlift_cam = d_bev_pool(dbev_cam, lift_feat_cam, geom_idx, bev_shape)  # (1,D,C_ctx,H,W)
        dlift_feats[idx] = dlift_cam[0]
    # 注意：dlift_feats 为 (B*N, D, C_ctx, Hf, Wf)

    # 4. Lift 反向
    features = caches['features']  # (B,N,Cf,Hf,Wf)
    lift_caches = caches['lift_caches']
    dfeatures_total = np.zeros_like(features)
    grads_lift_depth = {'w': None, 'b': None}
    grads_lift_context = {'w': None, 'b': None}
    for b in range(B):
        for n in range(N):
            idx = b * N + n
            d_lift_feat = dlift_feats[idx]  # (D, C_ctx, Hf, Wf)
            depth_logits, depth_prob, context, _, _ = lift_caches[idx]  # 注意 lift_cache 包含 depth_params, context_params 但这里不需要
            feat = features[b, n]
            # 需要重新获取 depth_params 和 context_params
            depth_params = model_params['lift_depth']
            context_params = model_params['lift_context']
            dfeat, depth_grads, context_grads = d_lift(
                d_lift_feat[None,...], depth_logits, depth_prob, context, feat[None,...],
                depth_params, context_params
            )
            dfeatures_total[b, n] = dfeat[0]
            # 累加梯度
            if grads_lift_depth['w'] is None:
                grads_lift_depth = depth_grads
                grads_lift_context = context_grads
            else:
                grads_lift_depth['w'] += depth_grads['w']
                grads_lift_depth['b'] += depth_grads['b']
                grads_lift_context['w'] += context_grads['w']
                grads_lift_context['b'] += context_grads['b']
            # ---- 额外处理深度损失梯度 ----
            if ddepth_list is not None:
                ddepth = ddepth_list[idx]  # (1, D, Hf, Wf)
                if ddepth is not None:
                    if ddepth.ndim ==3:
                        ddepth = ddepth[None,...]
                    if ddepth.shape[0] != 1:
                        raise ValueError(f"ddepth shape {ddepth.shape} expected batch size 1")
                     # 深度卷积反向：d_conv 需要输入 x (feat) 和输出梯度 ddepth
                    dfeat_extra, dw_extra, db_extra = d_conv(
                        ddepth, feat[None, ...], depth_params['w'], stride=1, padding=1
                    )
                    grads_lift_depth['w'] += dw_extra
                    grads_lift_depth['b'] += db_extra
                    dfeatures_total[b, n] += dfeat_extra[0]
    grads['lift_depth'] = grads_lift_depth
    grads['lift_context'] = grads_lift_context

    # 5. Backbone 反向
    backbone_caches = caches['backbone_caches']  # list of dicts
    dfeatures_flat = dfeatures_total.reshape(-1, *dfeatures_total.shape[2:])  # (B*N, Cf, Hf, Wf)
    backbone_grads = {}
    for i, cache in enumerate(backbone_caches):
        dfeat = dfeatures_flat[i][None, ...]
        _, grad_i = d_resnet18_features(dfeat, cache, model_params['backbone'])
        for k, v in grad_i.items():
            if k not in backbone_grads:
                backbone_grads[k] = v
            else:
                backbone_grads[k] += v
    grads['backbone'] = backbone_grads

    # 6. 深度损失梯度（如果有）
    if ddepth_list is not None:
        # ddepth_list 为每个相机的深度 logits 梯度，需累加到 lift_depth 梯度中
        # 由于 d_lift 已经计算了深度卷积梯度，但未包含来自深度损失的梯度，
        # 这里需要将 ddepth_list 传递给深度卷积反向，以更新深度卷积梯度
        # 但 d_lift 中 depth_params 的梯度是基于 dout_lift_feat 计算的，
        # 我们需要额外处理 ddepth_list
        # 这里为了简化，我们假设 ddepth_list 已经包含在 dout_lift_feat 中（通过损失函数组合），所以跳过
        pass

    return grads

# ============================
# 8. 参数初始化
# ============================

def init_conv_weight(shape, gain=1.0):
    fan_in = np.prod(shape[1:])
    std = gain / np.sqrt(fan_in)
    return np.random.randn(*shape) * std

def init_linear_weight(in_dim, out_dim, gain=1.0):
    std = gain / np.sqrt(in_dim)
    return np.random.randn(in_dim, out_dim) * std

def init_model_params(backbone_in=3, backbone_out=512,
                      depth_bins=41, context_channels=64,
                      bev_channels=64, bev_shape=(200,200),
                      num_classes=10):
    params = {}
    # Backbone (仅定义所需层)
    params['backbone'] = {}
    # stem
    params['backbone']['stem'] = {
        'conv1_w': init_conv_weight((64, backbone_in, 7, 7), gain=np.sqrt(2)),
        'conv1_b': np.zeros(64)
    }
    # stages
    for stage in range(4):
        C = 64  # 保持所有阶段通道数为 64
        for block in range(2):
            params['backbone'][f'stage{stage}'] = {
                'conv1_w': init_conv_weight((C, C, 3, 3), gain=np.sqrt(2)),
                'conv1_b': np.zeros(C),
                'conv2_w': init_conv_weight((C, C, 3, 3), gain=np.sqrt(2)),
                'conv2_b': np.zeros(C)
            }
    # Lift 输入通道改为 64
    params['lift_depth'] = {
        'w': init_conv_weight((depth_bins, backbone_out, 3, 3), gain=np.sqrt(2)),
        'b': np.zeros(depth_bins)
    }
    params['lift_context'] = {
        'w': init_conv_weight((context_channels, backbone_out, 3, 3), gain=np.sqrt(2)),
        'b': np.zeros(context_channels)
    }
    # BEV encoder
    params['bev_encoder'] = {
        'weights': [
            init_conv_weight((bev_channels, context_channels, 3, 3), gain=np.sqrt(2)),
            init_conv_weight((bev_channels, bev_channels, 3, 3), gain=np.sqrt(2))
        ],
        'biases': [np.zeros(bev_channels), np.zeros(bev_channels)]
    }
    # Head
    params['head'] = {
        'heatmap_w': init_conv_weight((num_classes, bev_channels, 3, 3), gain=1.0),
        'heatmap_b': np.zeros(num_classes),
        'reg_w': init_conv_weight((8, bev_channels, 3, 3), gain=1.0),
        'reg_b': np.zeros(8)
    }
    return params

# ============================
# 9. 测试函数
# ============================

if __name__ == "__main__":
    # 设置参数
    bev_shape = (200, 200)
    B, N, H, W = 2, 6, 256, 704
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

    # 随机输入
    images = np.random.randn(B, N, 3, H, W).astype(np.float32)

    # 生成随机几何索引（模拟）
    # 特征图尺寸：Hf=H//32=8, Wf=W//32=22, D=41
    Hf, Wf = H // 32, W // 32
    N_points = depth_bins * Hf * Wf
    geom_indices_list = []
    for b in range(B):
        cam_indices = []
        for n in range(N):
            # 随机索引 0~bev_shape[0]*bev_shape[1]-1
            idx = np.random.randint(0, bev_shape[0]*bev_shape[1], size=N_points)
            cam_indices.append(idx)
        geom_indices_list.append(cam_indices)

    # 前向
    heatmap, reg, depth_logits_list, bev_feat, caches = bev_forward(
        images, geom_indices_list, bev_shape, model_params
    )
    print("前向完成")
    print("heatmap shape:", heatmap.shape)
    print("reg shape:", reg.shape)
    print("depth_logits[0] shape:", depth_logits_list[0].shape)
    print("bev_feat shape:", bev_feat.shape)

    # 随机生成目标
    heatmap_gt = np.zeros_like(heatmap)
    heatmap_gt[0, 0, 100, 100] = 1
    reg_gt = np.random.randn(*reg.shape).astype(np.float32)
    depth_gt = np.random.randint(0, depth_bins, size=(B, Hf, Wf)).astype(np.int32)
    # 提取每个 batch 第一个相机的深度 logits，并堆叠为 (B, D, Hf, Wf)
    depth_logits_batch = np.stack([depth_logits_list[b * N] for b in range(B)], axis=0)  # (B, D, Hf, Wf)

    # 计算损失
    losses,loss_caches = compute_losses(heatmap, reg, heatmap_gt, reg_gt, depth_logits_batch, depth_gt)
    print("Losses:", losses)

    # 计算损失对输出的梯度
    dheatmap, dreg, ddepth = d_compute_losses(heatmap, reg, heatmap_gt, reg_gt, depth_logits_batch, depth_gt,loss_caches)
    print("dheatmap shape:", dheatmap.shape)
    print("dreg shape:", dreg.shape)
    print("ddepth shape:", ddepth.shape if ddepth is not None else None)

    # 反向传播
    grads = bev_backward(
        dheatmap, dreg, [ddepth] * (B*N),  # 每个相机相同的深度梯度（示例）
        caches, model_params, geom_indices_list, bev_shape
    )
    print("反向完成")
    print("梯度键:", grads.keys())

    # 检查梯度是否包含 nan
    has_nan = False
    for k, v in grads.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if np.any(np.isnan(vv)):
                    print(f"NaN found in {k}.{kk}")
                    has_nan = True
        else:
            if np.any(np.isnan(v)):
                print(f"NaN found in {k}")
                has_nan = True
    if not has_nan:
        print("所有梯度正常，无 NaN")

    print("测试完成")
