"""
Tacotron 2 简化版 - 单样本过拟合验证（完整前向+反向）
所有模块用 MMD 风格实现，纯函数式，无类。
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funcs import (
    linear, 
    conv,d_conv,
    silu, d_silu,
    softmax, d_softmax,
    layer_norm, d_layer_norm,
    matmul,
    adam_update,
	dropout,
	d_dropout
)
def post_net(x, params):
    """
    Post-Net: 5层卷积
    x: (batch, mel_len, mel_dim)
    返回: (out, caches)
        caches: 每层的 (conv_cache, pre_relu) 或 (conv_cache, None) 对于最后一层
    """
    batch, mel_len, mel_dim = x.shape
    x = x.transpose(0, 2, 1)  # (batch, mel_dim, mel_len)
    caches = []
    for l in range(5):
        # 卷积
        out, conv_cache = conv1d(x, params[f'post_{l}_w'], params[f'post_{l}_b'], 
                                 stride=1, padding='same')
        # 保存缓存
        if l < 4:
            pre_relu = out  # ReLU 前的值
            out = silu(out)
            caches.append((conv_cache, pre_relu))
        else:
            caches.append((conv_cache, None))  # 最后一层无激活
        x = out
    x = x.transpose(0, 2, 1)  # (batch, mel_len, mel_dim)
    return x, caches

def d_post_net(dout, caches, params):
    """
    Post-Net 反向
    """
    grads = {}
    dx = dout.transpose(0, 2, 1)  # (batch, mel_dim, mel_len)
    num_layers = len(caches)
    for l in range(num_layers - 1, -1, -1):
        conv_cache, pre_relu = caches[l]
        # 如果是最后一层
        if l == num_layers - 1:
            dx, dw, db = d_conv1d(dx, conv_cache)
        else:
            # ReLU 反向
            dx = d_silu(dx, pre_relu)  # 使用 ReLU 前的值
            dx, dw, db = d_conv1d(dx, conv_cache)
        grads[f'post_{l}_w'] = dw
        grads[f'post_{l}_b'] = db
    dx = dx.transpose(0, 2, 1)  # (batch, mel_len, mel_dim)
    return dx, grads
# ============================================================
# 1. 一维卷积（基于 MMD 的二维卷积）
# ============================================================
# ============================================================
# 1. 一维卷积（基于 MMD 的二维卷积）
# ============================================================
def conv1d(x, w, b=None, stride=1, padding='same'):
    """
    一维卷积，使用 MMD 的 conv 实现，手动处理填充
    x: (batch, in_channels, length)
    w: (out_channels, in_channels, kernel_size)
    b: (out_channels,)
    stride: int
    padding: 'same' 或 0（整数）
    """
    batch, C, L = x.shape
    out_C, in_C, k = w.shape
    if padding == 'same':
        pad = k // 2
    else:
        pad = padding  # 假设 padding 是整数
    # 手动填充宽度方向
    if pad > 0:
        x_padded = np.pad(x, ((0,0), (0,0), (pad, pad)), mode='constant')
    else:
        x_padded = x
    # 转换为二维：高度为1
    x_2d = x_padded[:, :, None, :]  # (batch, C, 1, L+2*pad)
    w_2d = w[:, :, None, :]  # (out_C, in_C, 1, k)
    # 调用 conv，padding=0，stride=stride
    out, _, (H_out, W_out) = conv(x_2d, w_2d, b, stride=stride, padding=0)
    # 去掉高度维度
    out = out[:, :, 0, :]
    # 缓存需要包含 x_2d, w_2d, b, stride, pad, x_padded, H_out, W_out
    cache = (x_2d, w_2d, b, stride, pad, x_padded, H_out, W_out)
    return out, cache
def d_conv1d(dout, cache):
    """
    一维卷积反向
    dout: (batch, out_C, L_out)
    cache: 来自 conv1d 的缓存
    """
    # 解包缓存
    x_2d, w_2d, b, stride, pad, x_padded, H_out, W_out = cache
    # 将 dout 扩展为二维
    dout_2d = dout[:, :, None, :]  # (batch, out_C, 1, L_out)
    # 调用 d_conv，注意 x 是 x_2d，x_pad 是 x_padded（已经是二维的）
    # d_conv 的 x_pad 参数应该是填充后的张量，我们传入 x_padded 但需要是二维的，x_padded 是 (batch, C, L+2*pad) 还需要加高度维度
    # 实际上 x_padded 是三维的 (batch, C, L+2*pad)，我们需要将其转为二维 (batch, C, 1, L+2*pad)
    x_padded_2d = x_padded[:, :, None, :]
    dx_2d, dw_2d, db = d_conv(
        dout_2d,          # dout
        x_2d,             # x
        w_2d,             # w
        stride=stride,    # stride
        padding=0,        # padding (因为已经手动填充)
        x_pad=x_padded_2d,# x_pad (填充后的张量)
        H_out=H_out,      # H_out
        W_out=W_out       # W_out
    )
    # 去掉高度维度
    dx = dx_2d[:, :, 0, :]  # (batch, in_C, L) 注意宽度是原始长度 L，但 dx_2d 的宽度是 L+2*pad，我们需要裁剪填充
    # 由于我们手动填充了宽度，反向传播的 dx 应该只有原始宽度部分，填充部分的梯度应为0
    # 所以我们需要裁剪掉填充部分
    if pad > 0:
        dx = dx[:, :, pad:-pad]
    dw = dw_2d[:, :, 0, :]  # (out_C, in_C, k)
    return dx, dw, db
def d_linear(dout, cache):
    """
    dout: (..., out_dim) 与 linear 输出形状一致
    cache: (x, w, b) 来自 linear 前向
    返回: dx (..., in_dim), dw (in_dim, out_dim), db (out_dim,)
    """
    x, w, b = cache
    # 展平前几维以便矩阵运算
    orig_shape = x.shape
    if x.ndim > 2:
        x_flat = x.reshape(-1, x.shape[-1])
        dout_flat = dout.reshape(-1, dout.shape[-1])
    else:
        x_flat = x
        dout_flat = dout
    
    dw = x_flat.T @ dout_flat
    db = dout_flat.sum(axis=0) if b is not None else None
    dx = dout_flat @ w.T
    
    if x.ndim > 2:
        dx = dx.reshape(orig_shape)
    return dx, dw, db
# ============================================================
# 1. 旋转位置编码 (RoPE)
# ============================================================
def get_rotary_embedding(seq_len, head_dim, base=10000):
    assert head_dim % 2 == 0
    i = np.arange(0, head_dim, 2)[None, :]
    theta = 1.0 / (base ** (i / head_dim))
    pos = np.arange(seq_len)[:, None]
    angles = pos * theta
    cos_emb = np.concatenate([np.cos(angles), np.cos(angles)], axis=1)
    sin_emb = np.concatenate([np.sin(angles), np.sin(angles)], axis=1)
    return cos_emb.astype(np.float32), sin_emb.astype(np.float32)

def apply_rotary(q, k, cos_emb, sin_emb):
    d = q.shape[-1] // 2
    q1, q2 = q[..., :d], q[..., d:]
    k1, k2 = k[..., :d], k[..., d:]
    cos = cos_emb[None, None, :, :d]
    sin = sin_emb[None, None, :, :d]
    q_rot = np.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    k_rot = np.concatenate([k1 * cos - k2 * sin, k1 * sin + k2 * cos], axis=-1)
    return q_rot, k_rot

def d_apply_rotary(dq_rot, dk_rot, q, k, cos_emb, sin_emb):
    d = q.shape[-1] // 2
    cos = cos_emb[None, None, :, :d]
    sin = sin_emb[None, None, :, :d]
    dq1_rot, dq2_rot = dq_rot[..., :d], dq_rot[..., d:]
    dk1_rot, dk2_rot = dk_rot[..., :d], dk_rot[..., d:]
    dq1 = dq1_rot * cos + dq2_rot * sin
    dq2 = -dq1_rot * sin + dq2_rot * cos
    dk1 = dk1_rot * cos + dk2_rot * sin
    dk2 = -dk1_rot * sin + dk2_rot * cos
    return np.concatenate([dq1, dq2], axis=-1), np.concatenate([dk1, dk2], axis=-1)

# ============================================================
# 2. 数据准备（单样本模拟）
# ============================================================
def create_sample(text="Hello world.", mel_len=50, mel_dim=80):
    """创建单样本：文本 + 随机梅尔频谱"""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    text_ids = np.array([[char2idx[c] for c in text]], dtype=np.int32)
    mel = np.random.randn(1, mel_len, mel_dim).astype(np.float32)
#    import pdb;pdb.set_trace()
    # ---- 改为全零 ----
    mel = 3.0+np.zeros((1, mel_len, mel_dim), dtype=np.float32)
    return {
        'text_ids': text_ids,
        'mel': mel,
        'vocab_size': len(chars),
        'char2idx': char2idx,
        'text': text,
    }
def load_sample_from_dataset(npz_path='tts_data.npz', sample_idx=1):
    """
    从提取好的 tts_data.npz 中加载一个样本
    """
    data = np.load(npz_path, allow_pickle=True)
    samples = data['samples'].tolist()
    char2idx = data['char2idx'].item()
    idx2char = data['idx2char'].item()
    vocab_size = int(data['vocab_size'])
    
    sample = samples[sample_idx]
    text_ids = np.array([sample['text_ids']], dtype=np.int32)
    mel = sample['mel'][None, :, :]  # 加 batch 维度 (1, T, 80)
    # ---- 归一化到 [-1, 1] ----
    mel_min = -80.0   # log-mel 的最小值（通常为 -80）
    mel_max = 0.0     # log-mel 的最大值（通常为 0）
    mel = (mel - mel_min) / (mel_max - mel_min) * 2.0 - 1.0
    # 现在 mel 的范围在 [-1, 1] 之间
    print(f"✅ 加载样本 {sample_idx+1}/{len(samples)}")
    print(f"  文本: {sample['text']}")
    print(f"  文本长度: {len(sample['text_ids'])}")
    print(f"  梅尔频谱帧数: {mel.shape[1]}, 维度: {mel.shape[2]}")
    
    return {
        'text_ids': text_ids,
        'mel': mel,
        'vocab_size': vocab_size,
        'char2idx': char2idx,
        'text': sample['text'],
        'idx2char': idx2char,
    }
# ============================================================
# 3. 文本编码器（Transformer Encoder，双向，RoPE）
# ============================================================
def encoder(params, text_ids):
    batch, seq_len = text_ids.shape
    embed_dim = params['embed_dim']
    vocab_size = params['vocab_size']
    num_heads = params['num_heads']
    head_dim = embed_dim // num_heads

    # 词嵌入
    one_hot = np.zeros((batch, seq_len, vocab_size), dtype=np.float32)
    one_hot[np.arange(batch)[:, None], np.arange(seq_len)[None, :], text_ids] = 1.0
    x, embed_cache = linear(one_hot, params['embed_weight'], None)
    caches = {'embed_cache': embed_cache, 'one_hot': one_hot}
    for l in range(params['num_encoder_layers']):
#        import pdb;pdb.set_trace()
        x_in = x
        # Q, K, V
        Q, q_cache = linear(x, params[f'enc_{l}_w_q'], params[f'enc_{l}_b_q'])
        K, k_cache = linear(x, params[f'enc_{l}_w_k'], params[f'enc_{l}_b_k'])
        V, v_cache = linear(x, params[f'enc_{l}_w_v'], params[f'enc_{l}_b_v'])

        Q = Q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        # RoPE
        cos_emb = params['cos_emb'][:seq_len, :]
        sin_emb = params['sin_emb'][:seq_len, :]
        Q_rot, K_rot = apply_rotary(Q, K, cos_emb, sin_emb)

        # 注意力分数
        scores = np.matmul(Q_rot, K_rot.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        attn, softmax_cache = softmax(scores, axis=-1)
        attn_out = np.matmul(attn, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        attn_out, attn_linear_cache = linear(attn_out, params[f'enc_{l}_w_o'], params[f'enc_{l}_b_o'])

        x = x + attn_out
        x, ln1_cache = layer_norm(x, params[f'enc_{l}_ln1_gamma'], params[f'enc_{l}_ln1_beta'])
        # FFN
        ff_h, ff1_cache = linear(x, params[f'enc_{l}_ff_w1'], params[f'enc_{l}_ff_b1'])
        ff_h_input = ff_h
        ff_h = silu(ff_h)
        ff_out, ff2_cache = linear(ff_h, params[f'enc_{l}_ff_w2'], params[f'enc_{l}_ff_b2'])
        x = x + ff_out
        x, ln2_cache = layer_norm(x, params[f'enc_{l}_ln2_gamma'], params[f'enc_{l}_ln2_beta'])

        caches[f'enc_{l}'] = {
            'x_in': x_in,
            'q_cache': q_cache, 'k_cache': k_cache, 'v_cache': v_cache,
            'Q': Q, 'K': K, 'V': V,
            'Q_rot': Q_rot, 'K_rot': K_rot,
            'scores': scores,
            'softmax_cache': softmax_cache,
            'attn': attn,
            'attn_linear_cache': attn_linear_cache,
            'ln1_cache': ln1_cache,
            'ff1_cache': ff1_cache,
            'ff2_cache': ff2_cache,
            'ln2_cache': ln2_cache,
            'cos_emb': cos_emb, 'sin_emb': sin_emb,
            'ff_h_input':ff_h_input,
        }
    caches['encoder_output'] = x
    return x, caches

def d_encoder(dout, caches, params):
    grads = {}
    dx = dout
    num_layers = params['num_encoder_layers']

    for l in range(num_layers - 1, -1, -1):
        layer = caches[f'enc_{l}']
        x_in = layer['x_in']
        Q = layer['Q']; K = layer['K']; V = layer['V']
        Q_rot = layer['Q_rot']; K_rot = layer['K_rot']
        scores = layer['scores']
        attn = layer['attn']
        softmax_cache = layer['softmax_cache']
        attn_linear_cache = layer['attn_linear_cache']
        ln1_cache = layer['ln1_cache']
        ff1_cache = layer['ff1_cache']
        ff2_cache = layer['ff2_cache']
        ln2_cache = layer['ln2_cache']
        cos_emb = layer['cos_emb']; sin_emb = layer['sin_emb']
        num_heads = params['num_heads']; head_dim = params['embed_dim'] // num_heads
        embed_dim = params['embed_dim']

        # LN2 反向
        dx, dgamma2, dbeta2 = d_layer_norm(dx, ln2_cache)
        grads[f'enc_{l}_ln2_gamma'] = dgamma2
        grads[f'enc_{l}_ln2_beta'] = dbeta2
        # FFN 反向
        dx_ff, dw_ff2, db_ff2 = d_linear(dx, ff2_cache)
        grads[f'enc_{l}_ff_w2'] = dw_ff2
        grads[f'enc_{l}_ff_b2'] = db_ff2
        dh = d_silu(dx_ff, layer['ff_h_input'])
        dx_ff, dw_ff1, db_ff1 = d_linear(dh, ff1_cache)
        grads[f'enc_{l}_ff_w1'] = dw_ff1
        grads[f'enc_{l}_ff_b1'] = db_ff1
        dx = dx_ff + dx

        # LN1 反向
        dx, dgamma1, dbeta1 = d_layer_norm(dx, ln1_cache)
        grads[f'enc_{l}_ln1_gamma'] = dgamma1
        grads[f'enc_{l}_ln1_beta'] = dbeta1

        # 注意力反向
        dx_attn, dw_o, db_o = d_linear(dx, attn_linear_cache)
        grads[f'enc_{l}_w_o'] = dw_o
        grads[f'enc_{l}_b_o'] = db_o

        batch, seq_len, _ = x_in.shape
        dx_attn = dx_attn.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        dV = np.matmul(attn.transpose(0, 1, 3, 2), dx_attn)
        dattn = np.matmul(dx_attn, V.transpose(0, 1, 3, 2))

        dscores = d_softmax(dattn, softmax_cache)
        dscores = dscores / np.sqrt(head_dim)

        dK_rot = np.matmul(dscores.transpose(0, 1, 3, 2), Q_rot)
        dQ_rot = np.matmul(dscores, K_rot)

        dQ, dK = d_apply_rotary(dQ_rot, dK_rot, Q, K, cos_emb, sin_emb)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)

        dx_q, dw_q, db_q = d_linear(dQ, layer['q_cache'])
        dx_k, dw_k, db_k = d_linear(dK, layer['k_cache'])
        dx_v, dw_v, db_v = d_linear(dV, layer['v_cache'])

        grads[f'enc_{l}_w_q'] = dw_q
        grads[f'enc_{l}_b_q'] = db_q
        grads[f'enc_{l}_w_k'] = dw_k
        grads[f'enc_{l}_b_k'] = db_k
        grads[f'enc_{l}_w_v'] = dw_v
        grads[f'enc_{l}_b_v'] = db_v

        dx = dx_q + dx_k + dx_v + dx

    # 词嵌入反向
    one_hot = caches['one_hot']
    d_embed_weight = np.matmul(one_hot.transpose(0, 2, 1), dx).sum(axis=0)
    grads['embed_weight'] = d_embed_weight

    return grads
# ============================================================
# 1. 因果掩码生成
# ============================================================
def causal_mask(seq_len):
    """
    生成因果掩码（下三角矩阵）
    形状: (seq_len, seq_len)
    值: 0 表示允许，-1e9 表示屏蔽
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    return mask.astype(np.float32)
def location_sensitive_attention(params, query, keys, values, location_features):
    """
    多帧 query 版本，使用向量位置权重
    query: (batch, seq_len, embed_dim)
    keys: (batch, T_enc, embed_dim)
    values: (batch, T_enc, embed_dim)
    location_features: (batch, seq_len, T_enc)
    返回: context, attn, updated_location_features, cache
    """
    batch, seq_len, embed_dim = query.shape
    _, T_enc, _ = keys.shape

    # 1. 投影 query 和 keys
    q_proj = np.matmul(query, params['attn_w_q']) + params['attn_b_q']  # (batch, seq_len, embed_dim)
    k_proj = np.matmul(keys, params['attn_w_k']) + params['attn_b_k']  # (batch, T_enc, embed_dim)

    # 2. 位置特征投影（向量版本）
    # location_features: (batch, seq_len, T_enc) -> 扩展为 (batch, seq_len, T_enc, 1)
    # attn_w_loc: (1, embed_dim) 或 (embed_dim, embed_dim)
    # 推荐使用 (1, embed_dim)，这样 location_features 与 attn_w_loc 相乘后得到 (batch, seq_len, T_enc, embed_dim)
#    loc_proj = np.matmul(location_features[:, :, :, None], params['attn_w_loc'])  # (batch, seq_len, T_enc, embed_dim)
#    loc_proj = loc_proj.squeeze(-2)  # 去掉多余的维度，但注意 squeeze 可能会出错，我们明确使用 reshape
    # 更清晰的方式：
    loc_proj = np.matmul(location_features[:, :, :, None], params['attn_w_loc'])  # (batch, seq_len, T_enc, embed_dim)
    loc_proj = loc_proj.reshape(batch, seq_len, T_enc, embed_dim)  # 确保形状正确

    # 3. 计算注意力分数
    # q_proj: (batch, seq_len, embed_dim) -> 扩展为 (batch, seq_len, 1, embed_dim)
    scores = np.sum(q_proj[:, :, None, :] * k_proj[:, None, :, :], axis=-1)  # (batch, seq_len, T_enc)
    scores = scores / np.sqrt(embed_dim)  # 缩放

    # 加上位置特征贡献（对 embed_dim 维度求和）
    scores = scores + np.sum(loc_proj, axis=-1)  # (batch, seq_len, T_enc)

    # 4. softmax
    attn,softmax_cache = softmax(scores, axis=-1)  # (batch, seq_len, T_enc)

    # 5. 上下文向量
    context = np.matmul(attn[:, :, None, :], values[:, None, :, :]).squeeze(2)  # (batch, seq_len, embed_dim)

    # 6. 更新位置特征
    location_features = location_features + attn

    # 缓存
    cache = {
        'query': query,
        'keys': keys,
        'values': values,
        'location_features': location_features,
        'attn': attn,
        'softmax_cache':softmax_cache,
        'scores': scores,
        'q_proj': q_proj,
        'k_proj': k_proj,
        'loc_proj': loc_proj,
        'embed_dim': embed_dim,
        'T_enc': T_enc,
        'seq_len': seq_len,
    }
    return context, attn, location_features, cache
def d_location_sensitive_attention(dcontext, dattn, cache, params):
    """
    向量版本的反向传播
    dcontext: (batch, seq_len, embed_dim)
    dattn: (batch, seq_len, T_enc)
    cache: 前向缓存
    params: 模型参数
    """
    query = cache['query']
    keys = cache['keys']
    values = cache['values']
    location_features = cache['location_features']
    attn = cache['attn']
    softmax_cache = cache['softmax_cache']
    scores = cache['scores']
    q_proj = cache['q_proj']
    k_proj = cache['k_proj']
    loc_proj = cache['loc_proj']
    embed_dim = cache['embed_dim']
    T_enc = cache['T_enc']
    seq_len = cache['seq_len']
    batch, seq_len, embed_dim = query.shape

    # 1. 计算 attn 的总梯度
    dattn_total = dattn + np.sum(dcontext[:, :, None, :] * values[:, None, :, :], axis=-1)  # (batch, seq_len, T_enc)

    # 2. softmax 反向
    dscores = d_softmax(dattn_total, softmax_cache)  # (batch, seq_len, T_enc)
    dscores = dscores / np.sqrt(embed_dim)

    # 3. 对 q_proj, k_proj, loc_proj 的梯度
    # scores = sum(q_proj * k_proj) + sum(loc_proj)
    dq_exp = dscores[:, :, :, None] * k_proj[:, None, :, :]  # (batch, seq_len, T_enc, embed_dim)
    dk_proj = np.sum(dscores[:, :, :, None] * q_proj[:, :, None, :], axis=1)  # (batch, T_enc, embed_dim)
    dloc_proj = dscores[:, :, :, None]  # (batch, seq_len, T_enc, embed_dim)

    # 4. query 的梯度
    dquery = np.sum(dq_exp, axis=2)  # (batch, seq_len, embed_dim)

    # 5. keys 的梯度
    dkeys = dk_proj  # (batch, T_enc, embed_dim)

    # 6. values 的梯度
    dvalues = np.sum(attn[:, :, None, :] * dcontext[:, :, None, :], axis=1)  # (batch, T_enc, embed_dim)

    # 7. location_features 的梯度
    # loc_proj = location_features @ attn_w_loc（其中 attn_w_loc 形状为 (1, embed_dim) 或 (embed_dim, embed_dim)）
    # 这里假设 attn_w_loc 是 (1, embed_dim)，则
    # dlocation = dloc_proj @ attn_w_loc.T  (将 embed_dim 维度的梯度投影回 T_enc)
    # 但 dloc_proj 是 (batch, seq_len, T_enc, embed_dim)
    # 我们计算 dlocation = np.sum(dloc_proj * attn_w_loc, axis=-1) 如果 attn_w_loc 是 (1, embed_dim)
    # 更通用：如果 attn_w_loc 是 (embed_dim, embed_dim)，则用矩阵乘法
    # 这里为了兼容，假设 attn_w_loc 是 (1, embed_dim)
    dlocation = np.sum(dloc_proj * params['attn_w_loc'], axis=-1)  # (batch, seq_len, T_enc)

    # 8. 参数梯度
    dw_q = np.matmul(query.transpose(0, 2, 1), dquery).sum(axis=0)  # (embed_dim, embed_dim)
    db_q = dquery.sum(axis=(0, 1))  # (embed_dim,)
    dw_k = np.matmul(keys.transpose(0, 2, 1), dkeys).sum(axis=0)  # (embed_dim, embed_dim)
    db_k = dkeys.sum(axis=(0, 1))  # (embed_dim,)

    # attn_w_loc: (1, embed_dim)
    # loc_proj = location_features @ attn_w_loc + attn_b_loc
    # 所以 dw_loc = (location_features 与 dloc_proj 的乘积) 在 batch, seq_len 上求和
    # location_features: (batch, seq_len, T_enc) -> 扩展为 (batch, seq_len, T_enc, 1)
    dw_loc = np.matmul(location_features[:, :, :, None].transpose(0, 1, 3, 2), dloc_proj).sum(axis=(0, 1))  # (1, embed_dim)
    db_loc = dloc_proj.sum(axis=(0, 1, 2))  # (embed_dim,)

    grads = {
        'attn_w_q': dw_q,
        'attn_b_q': db_q,
        'attn_w_k': dw_k,
        'attn_b_k': db_k,
        'attn_w_loc': dw_loc,
        'attn_b_loc': db_loc,
    }

    return dquery, dkeys, dvalues, dlocation, grads
# ============================================================
# 3. 完整的Transformer解码器
# ============================================================
def decoder(params, encoder_output, mel_targets=None, teacher_forcing=True, max_len=200):
    """
    Transformer 解码器（完整版）
    支持训练（Teacher Forcing）和推理（自回归）
    使用多帧 query 的 Location-Sensitive Attention
    """
    batch, T_enc, embed_dim = encoder_output.shape
    mel_dim = params['mel_dim']
    num_layers = params['decoder_layers']
    num_heads = params['num_heads']
    dropout_rate = params.get('dropout_rate', 0.1)

    # 因果掩码生成函数
    def causal_mask(seq_len):
        return np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9

    if teacher_forcing and mel_targets is not None:
        # =================== 训练模式 ===================
        mel_len = mel_targets.shape[1] - 1
        inputs = mel_targets[:, :-1, :]  # (batch, mel_len, mel_dim)
        # ---- Pre-Net ----
        pre_out, pre_cache_1 = linear(inputs, params['decoder_pre_w1'], params['decoder_pre_b1'])
        pre_out, dropout_cache1 = dropout(pre_out, dropout_rate)
        pre_out, pre_cache_2 = linear(pre_out, params['decoder_pre_w2'], params['decoder_pre_b2'])
        pre_out, dropout_cache2 = dropout(pre_out, dropout_rate)
        # pre_out: (batch, mel_len, embed_dim)

        # ---- 因果掩码 ----
        mask = causal_mask(mel_len)

        # ---- 初始化位置特征 ----
        location_features = np.zeros((batch, mel_len, T_enc), dtype=np.float32)

        # ---- Transformer 层 ----
        x = pre_out
        layer_caches = {}
        attn_list = []

        for l in range(num_layers):
            # 1. 自注意力（带因果掩码）
            attn_out, attn_self_cache = self_attention(params, x, mask, l)
            x = x + attn_out
            x, ln1_cache = layer_norm(x, params[f'dec_{l}_self_ln_gamma'], params[f'dec_{l}_self_ln_beta'])

            # 2. 交叉注意力（多帧 query）
            query, q_cross_cache = linear(x, params[f'dec_{l}_cross_w_q'], params[f'dec_{l}_cross_b_q'])
            context, attn_cross, location_features, attn_cross_cache = location_sensitive_attention(
                params, query, encoder_output, encoder_output, location_features
            )
            x = x + context
            x, ln2_cache = layer_norm(x, params[f'dec_{l}_cross_ln_gamma'], params[f'dec_{l}_cross_ln_beta'])

            # 3. FFN
            ff_h, ff1_cache = linear(x, params[f'dec_{l}_ff_w1'], params[f'dec_{l}_ff_b1'])
            ff_h_input = ff_h
            ff_h = silu(ff_h)
            ff_out, ff2_cache = linear(ff_h, params[f'dec_{l}_ff_w2'], params[f'dec_{l}_ff_b2'])
            x = x + ff_out
            x, ln3_cache = layer_norm(x, params[f'dec_{l}_ff_ln_gamma'], params[f'dec_{l}_ff_ln_beta'])

            # 保存该层缓存
            layer_caches[f'layer_{l}'] = {
                'attn_self_cache': attn_self_cache,
                'ln1_cache': ln1_cache,
                'q_cross_cache': q_cross_cache,
                'attn_cross_cache': attn_cross_cache,
                'ln2_cache': ln2_cache,
                'ff1_cache': ff1_cache,
                'ff2_cache': ff2_cache,
                'ff_h_input':ff_h_input,
                'ln3_cache': ln3_cache,
                'attn_cross': attn_cross,
              
            }
            attn_list.append(attn_cross)

        # ---- 输出层 ----
        pred_mel, mel_cache = linear(x, params['decoder_w_out'], params['decoder_b_out'])  # (batch, mel_len, mel_dim)

        # ---- Post-Net ----
        pred_mel_post, post_caches = post_net(pred_mel, params)
        pred_mel = pred_mel + pred_mel_post  # 残差连接

        # ---- 缓存所有中间结果（用于反向传播） ----
        cache = {
            'mode': 'train',
            'outputs': pred_mel,
            'attn_list': attn_list,
            'layer_caches': layer_caches,
            'mel_cache': mel_cache,
            'pre_cache_1': pre_cache_1,
            'pre_cache_2': pre_cache_2,
            'dropout_cache1': dropout_cache1,
            'dropout_cache2': dropout_cache2,
            'post_caches': post_caches,
            'mask': mask,
            'location_features': location_features,
            'mel_targets': mel_targets,
            'T_enc':T_enc,
        }
        return pred_mel, cache

    else:
        # =================== 推理模式（自回归） ===================
        outputs = []
        attn_list = []
        history = []  # 存储每帧的 Pre-Net 输出（embed_dim）
        location_features = np.zeros((batch, 1, T_enc), dtype=np.float32)  # 每步更新

        for t in range(max_len):
            # 当前步的输入
            if t == 0:
                prev_output = np.zeros((batch,mel_dim), dtype=np.float32)
            else:
                prev_output = outputs[-1]  # 上一帧的预测

            # ---- Pre-Net ----
            pre_out, _ = linear(prev_output, params['decoder_pre_w1'], params['decoder_pre_b1'])
            pre_out,_ = dropout(pre_out, dropout_rate, training=False)
            pre_out, _ = linear(pre_out, params['decoder_pre_w2'], params['decoder_pre_b2'])
            pre_out,_ = dropout(pre_out, dropout_rate, training=False)
            # pre_out: (batch, embed_dim)
            # 添加到历史
            history.append(pre_out)  # (batch, embed_dim)
            x_seq = np.stack(history, axis=1)  # (batch, seq_len, embed_dim)
            cur_len = x_seq.shape[1]

            # ---- 因果掩码 ----
            mask = causal_mask(cur_len)

            # ---- Transformer 层 ----
            x = x_seq
            for l in range(num_layers):
                # 自注意力
                attn_out, _ = self_attention(params, x, mask, l)
                x = x + attn_out
                x, _ = layer_norm(x, params[f'dec_{l}_self_ln_gamma'], params[f'dec_{l}_self_ln_beta'])

                # 交叉注意力（只对最后一帧做）
                query = x[:, -1:, :]  # (batch, 1, embed_dim)
                query, _ = linear(query, params[f'dec_{l}_cross_w_q'], params[f'dec_{l}_cross_b_q'])
                context, attn_cross, location_features, _ = location_sensitive_attention(
                    params, query, encoder_output, encoder_output, location_features
                )
                # context: (batch, 1, embed_dim)，扩展到所有帧以便残差
                context_exp = np.tile(context, (1, cur_len, 1))
                x = x + context_exp
                x, _ = layer_norm(x, params[f'dec_{l}_cross_ln_gamma'], params[f'dec_{l}_cross_ln_beta'])

                # FFN
                ff_h, _ = linear(x, params[f'dec_{l}_ff_w1'], params[f'dec_{l}_ff_b1'])
                ff_h = silu(ff_h)
                ff_out, _ = linear(ff_h, params[f'dec_{l}_ff_w2'], params[f'dec_{l}_ff_b2'])
                x = x + ff_out
                x, _ = layer_norm(x, params[f'dec_{l}_ff_ln_gamma'], params[f'dec_{l}_ff_ln_beta'])

            # ---- 输出当前帧（x 的最后一帧） ----
            mel_frame, _ = linear(x[:, -1, :], params['decoder_w_out'], params['decoder_b_out'])
            outputs.append(mel_frame)
            attn_list.append(attn_cross)

            if t >= max_len - 1:
                break

        pred_mel = np.stack(outputs, axis=1)  # (batch, mel_len, mel_dim)

        # ---- Post-Net ----
        pred_mel_post, post_caches = post_net(pred_mel, params)
        pred_mel = pred_mel + pred_mel_post

        cache = {
            'mode': 'infer',
            'outputs': pred_mel,
            'attn_list': attn_list,
            'post_caches': post_caches,
        }
        return pred_mel, cache
# ============================================================
# 4. 自注意力函数
# ============================================================
def self_attention(params, x, mask, layer_idx):
    """
    自注意力（带因果掩码）
    """
    batch, seq_len, embed_dim = x.shape
    num_heads = params['num_heads']
    head_dim = embed_dim // num_heads

    Q, q_cache = linear(x, params[f'dec_{layer_idx}_self_w_q'], params[f'dec_{layer_idx}_self_b_q'])
    K, k_cache = linear(x, params[f'dec_{layer_idx}_self_w_k'], params[f'dec_{layer_idx}_self_b_k'])
    V, v_cache = linear(x, params[f'dec_{layer_idx}_self_w_v'], params[f'dec_{layer_idx}_self_b_v'])

    Q = Q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # RoPE
    cos_emb = params['decoder_cos_emb'][:seq_len, :]
    sin_emb = params['decoder_sin_emb'][:seq_len, :]
    Q_rot, K_rot = apply_rotary(Q, K, cos_emb, sin_emb)

    scores = np.matmul(Q_rot, K_rot.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    scores = scores + mask[None, None, :, :]
    attn, softmax_cache = softmax(scores, axis=-1)
    attn_out = np.matmul(attn, V)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
    attn_out, attn_linear_cache = linear(attn_out, params[f'dec_{layer_idx}_self_w_o'], params[f'dec_{layer_idx}_self_b_o'])

    cache = {
        'q_cache': q_cache,
        'k_cache': k_cache,
        'v_cache': v_cache,
        'softmax_cache': softmax_cache,
        'attn_linear_cache': attn_linear_cache,
        'Q': Q,
        'K': K,
        'V': V,
        'Q_rot': Q_rot,
        'K_rot': K_rot,
    }
    return attn_out, cache
def d_decoder(dout_mel, cache, params):
    """
    完整的 Transformer 解码器反向传播（匹配 decoder 前向）
    dout_mel: (batch, mel_len, mel_dim) 损失对最终输出的梯度
    cache: decoder 前向传播保存的缓存
    params: 模型参数
    返回: (grads, dx_encoder)
    """
    grads = {}
    # 解包缓存
    mel_targets = cache.get('mel_targets')
    layer_caches = cache['layer_caches']
    post_caches = cache['post_caches']
    mel_cache = cache['mel_cache']
    pre_cache_1 = cache['pre_cache_1']
    pre_cache_2 = cache['pre_cache_2']
    dropout_cache1 = cache['dropout_cache1']
    dropout_cache2 = cache['dropout_cache2']
    mask = cache['mask']
    batch, mel_len, mel_dim = dout_mel.shape
    embed_dim = params['embed_dim']
    num_layers = params['decoder_layers']
    dropout_rate = params.get('dropout_rate', 0.1)
    #T_enc = params.get('T_enc') or cache.get('T_enc', 8)  # 从缓存或参数获取
    T_enc = cache['T_enc']

    # ---- 1. Post-Net 反向 ----
    dout_post, post_grads = d_post_net(dout_mel, post_caches, params)
    grads.update(post_grads)
    # 残差：final = mel + post_net(mel)
    dx_mel = dout_mel + dout_post  # (batch, mel_len, mel_dim)

    # ---- 2. 输出层反向 ----
    dx, dw_out, db_out = d_linear(dx_mel, mel_cache)
    grads['decoder_w_out'] = dw_out
    grads['decoder_b_out'] = db_out
    # dx: (batch, mel_len, embed_dim)

    # ---- 3. 逐层反向（从最后一层到第一层） ----
    dx_encoder = np.zeros((batch, T_enc, embed_dim), dtype=np.float32)

    for l in range(num_layers - 1, -1, -1):
        layer = layer_caches[f'layer_{l}']

        # 3a. FFN 反向（使用 SiLU）
        dx, dgamma_ln3, dbeta_ln3 = d_layer_norm(dx, layer['ln3_cache'])
        grads[f'dec_{l}_ff_ln_gamma'] = dgamma_ln3
        grads[f'dec_{l}_ff_ln_beta'] = dbeta_ln3

        dx_ff, dw_ff2, db_ff2 = d_linear(dx, layer['ff2_cache'])
        grads[f'dec_{l}_ff_w2'] = dw_ff2
        grads[f'dec_{l}_ff_b2'] = db_ff2

        # SiLU 反向（使用保存的 ff_h_input）
        dx_ff = d_silu(dx_ff, layer['ff_h_input'])

        dx_ff, dw_ff1, db_ff1 = d_linear(dx_ff, layer['ff1_cache'])
        grads[f'dec_{l}_ff_w1'] = dw_ff1
        grads[f'dec_{l}_ff_b1'] = db_ff1

        dx = dx_ff + dx  # 残差

        # 3b. 交叉注意力反向
        dx, dgamma_ln2, dbeta_ln2 = d_layer_norm(dx, layer['ln2_cache'])
        grads[f'dec_{l}_cross_ln_gamma'] = dgamma_ln2
        grads[f'dec_{l}_cross_ln_beta'] = dbeta_ln2

        dquery, dw_q_cross, db_q_cross = d_linear(dx, layer['q_cross_cache'])
        grads[f'dec_{l}_cross_w_q'] = dw_q_cross
        grads[f'dec_{l}_cross_b_q'] = db_q_cross

        # 调用 d_location_sensitive_attention
        # 注意：这里 dattn 传零，因为 encoder_output 的梯度由 dkeys_attn + dvalues_attn 给出
        dquery_attn, dkeys_attn, dvalues_attn, dlocation_attn, attn_grads = d_location_sensitive_attention(
            dquery,
            np.zeros_like(layer['attn_cross']),
            layer['attn_cross_cache'],
            params
        )
        for k, v in attn_grads.items():
            if k not in grads:
                grads[k] = np.zeros_like(v)
            grads[k] += v
        # 累加编码器梯度（来自 keys 和 values）
        dx_encoder += dkeys_attn + dvalues_attn

        dx = dquery_attn + dx  # 残差

        # 3c. 自注意力反向
        dx, dgamma_ln1, dbeta_ln1 = d_layer_norm(dx, layer['ln1_cache'])
        grads[f'dec_{l}_self_ln_gamma'] = dgamma_ln1
        grads[f'dec_{l}_self_ln_beta'] = dbeta_ln1

        # 从缓存中取出 self_attention 的缓存
        self_cache = layer['attn_self_cache']
        # self_cache 包含: q_cache, k_cache, v_cache, softmax_cache, attn_linear_cache,
        # Q, K, V, Q_rot, K_rot

        # 自注意力输出线性层反向
        dx_self, dw_o_self, db_o_self = d_linear(dx, self_cache['attn_linear_cache'])
        grads[f'dec_{l}_self_w_o'] = dw_o_self
        grads[f'dec_{l}_self_b_o'] = db_o_self

        # 恢复形状 (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        batch, seq_len, embed_dim = dx_self.shape
        num_heads = params['num_heads']
        head_dim = embed_dim // num_heads
        dx_self = dx_self.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        # 从缓存中获取 V 和 attn（softmax 输出）
        V = self_cache['V']
        attn = self_cache['softmax_cache']  # softmax 输出

        # 计算 V 和 attn 的梯度
        dV = np.matmul(attn.transpose(0, 1, 3, 2), dx_self)
        dattn = np.matmul(dx_self, V.transpose(0, 1, 3, 2))

        # softmax 反向
        dscores = d_softmax(dattn, self_cache['softmax_cache'])
        dscores = dscores / np.sqrt(head_dim)

        # 应用因果掩码（屏蔽位置的梯度置零）
        # mask 形状是 (seq_len, seq_len)，需要扩展到 (batch, heads, seq_len, seq_len)
        mask_matrix = mask[:seq_len, :seq_len]  # (seq_len, seq_len)
        dscores = dscores * (mask_matrix == 0)  # 允许的位置保留，屏蔽的位置置零

        # 计算 Q_rot 和 K_rot 的梯度
        Q_rot = self_cache['Q_rot']
        K_rot = self_cache['K_rot']
        dK_rot = np.matmul(dscores.transpose(0, 1, 3, 2), Q_rot)
        dQ_rot = np.matmul(dscores, K_rot)

        # RoPE 反向
        cos_emb = params['decoder_cos_emb'][:seq_len, :]
        sin_emb = params['decoder_sin_emb'][:seq_len, :]
        Q = self_cache['Q']
        K = self_cache['K']
        dQ, dK = d_apply_rotary(dQ_rot, dK_rot, Q, K, cos_emb, sin_emb)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)

        # Q, K, V 的线性层反向
        dx_q, dw_q, db_q = d_linear(dQ, self_cache['q_cache'])
        dx_k, dw_k, db_k = d_linear(dK, self_cache['k_cache'])
        dx_v, dw_v, db_v = d_linear(dV, self_cache['v_cache'])

        grads[f'dec_{l}_self_w_q'] = dw_q
        grads[f'dec_{l}_self_b_q'] = db_q
        grads[f'dec_{l}_self_w_k'] = dw_k
        grads[f'dec_{l}_self_b_k'] = db_k
        grads[f'dec_{l}_self_w_v'] = dw_v
        grads[f'dec_{l}_self_b_v'] = db_v

        # 残差：x = x + attn_out，所以 dx 需要累加
        dx = dx_q + dx_k + dx_v + dx

    # ---- 4. Pre-Net 反向 ----
    # dx 现在是 Pre-Net 输出的梯度（即第一层输入）
    # 反向 Pre-Net 第二层
    dx = d_dropout(dx, dropout_cache2)
    dx, dw_pre2, db_pre2 = d_linear(dx, pre_cache_2)
    grads['decoder_pre_w2'] = dw_pre2
    grads['decoder_pre_b2'] = db_pre2

    # 反向 Pre-Net 第一层
    dx = d_dropout(dx, dropout_cache1)
    dx, dw_pre1, db_pre1 = d_linear(dx, pre_cache_1)
    grads['decoder_pre_w1'] = dw_pre1
    grads['decoder_pre_b1'] = db_pre1

    # ---- 5. 返回 ----
    return grads, dx_encoder
# ============================================================
# 6. 完整 TTS 模型（前向 + 反向）
# ============================================================
def tts(params, text_ids, mel_targets=None, teacher_forcing=True,max_len=200):
    encoder_output, enc_caches = encoder(params, text_ids)
    pred_mel, dec_caches = decoder(params, encoder_output, mel_targets, teacher_forcing,max_len)
    return pred_mel, {'encoder': enc_caches, 'decoder': dec_caches}

def d_tts(dout_mel, caches, params):
    # 先反向解码器
    dec_grads, dx_encoder = d_decoder(dout_mel, caches['decoder'], params)
    # 再反向编码器
    enc_grads = d_encoder(dx_encoder, caches['encoder'], params)
    grads = {**dec_grads, **enc_grads}
    return grads

# ============================================================
# 7. 参数初始化
# ============================================================
def init_tts_params(vocab_size, embed_dim=128, num_heads=8, num_encoder_layers=3, num_decoder_layers=3,mel_dim=80, max_seq_len=200,dropout_rate=0.1,post_net_channels=512):
    head_dim = embed_dim // num_heads
    cos_emb, sin_emb = get_rotary_embedding(max_seq_len, head_dim)

    params = {
        'embed_dim': embed_dim,
        'vocab_size': vocab_size,
        'num_heads': num_heads,
        'num_encoder_layers': num_encoder_layers,
        'mel_dim': mel_dim,
        'cos_emb': cos_emb,
        'sin_emb': sin_emb,
        'embed_weight': np.random.randn(vocab_size, embed_dim) * 0.1,
    }

    # 编码器各层
    for l in range(num_encoder_layers):
        params[f'enc_{l}_w_q'] = np.random.randn(embed_dim, embed_dim) * 0.1
        params[f'enc_{l}_b_q'] = np.zeros(embed_dim)
        params[f'enc_{l}_w_k'] = np.random.randn(embed_dim, embed_dim) * 0.1
        params[f'enc_{l}_b_k'] = np.zeros(embed_dim)
        params[f'enc_{l}_w_v'] = np.random.randn(embed_dim, embed_dim) * 0.1
        params[f'enc_{l}_b_v'] = np.zeros(embed_dim)
        params[f'enc_{l}_w_o'] = np.random.randn(embed_dim, embed_dim) * 0.1
        params[f'enc_{l}_b_o'] = np.zeros(embed_dim)
        params[f'enc_{l}_ln1_gamma'] = np.ones(embed_dim)
        params[f'enc_{l}_ln1_beta'] = np.zeros(embed_dim)
        params[f'enc_{l}_ln2_gamma'] = np.ones(embed_dim)
        params[f'enc_{l}_ln2_beta'] = np.zeros(embed_dim)
        params[f'enc_{l}_ff_w1'] = np.random.randn(embed_dim, embed_dim*4) * 0.1
        params[f'enc_{l}_ff_b1'] = np.zeros(embed_dim*4)
        params[f'enc_{l}_ff_w2'] = np.random.randn(embed_dim*4, embed_dim) * 0.1
        params[f'enc_{l}_ff_b2'] = np.zeros(embed_dim)




    # 解码器各层
    params['decoder_layers'] = num_decoder_layers
    params['num_heads'] = num_heads
    params['mel_dim'] = mel_dim
    params['dropout_rate'] = dropout_rate
    params['embed_dim'] = embed_dim

    # 位置编码
    head_dim = embed_dim // num_heads
    cos_emb, sin_emb = get_rotary_embedding(max_seq_len, head_dim)
    params['decoder_cos_emb'] = cos_emb
    params['decoder_sin_emb'] = sin_emb


    # Pre-Net (2层)
    params['decoder_pre_w1'] = np.random.randn(mel_dim, embed_dim) * 0.02
    params['decoder_pre_b1'] = np.zeros(embed_dim)
    params['decoder_pre_w2'] = np.random.randn(embed_dim, embed_dim) * 0.02
    params['decoder_pre_b2'] = np.zeros(embed_dim)

    # 输出层
    params['decoder_w_out'] = np.random.randn(embed_dim, mel_dim) * 0.02
    params['decoder_b_out'] = np.zeros(mel_dim)

    # Post-Net (5层卷积)
    for l in range(5):
        in_channels = mel_dim if l == 0 else post_net_channels
        out_channels = post_net_channels if l < 4 else mel_dim
        kernel_size = 5
        params[f'post_{l}_w'] = np.random.randn(out_channels, in_channels, kernel_size) * 0.02
        params[f'post_{l}_b'] = np.zeros(out_channels)

    # Transformer层
    for l in range(num_decoder_layers):
        # 自注意力
        params[f'dec_{l}_self_w_q'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'dec_{l}_self_b_q'] = np.zeros(embed_dim)
        params[f'dec_{l}_self_w_k'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'dec_{l}_self_b_k'] = np.zeros(embed_dim)
        params[f'dec_{l}_self_w_v'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'dec_{l}_self_b_v'] = np.zeros(embed_dim)
        params[f'dec_{l}_self_w_o'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'dec_{l}_self_b_o'] = np.zeros(embed_dim)
        params[f'dec_{l}_self_ln_gamma'] = np.ones(embed_dim)
        params[f'dec_{l}_self_ln_beta'] = np.zeros(embed_dim)

        # 交叉注意力
        params[f'dec_{l}_cross_w_q'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'dec_{l}_cross_b_q'] = np.zeros(embed_dim)
        params[f'dec_{l}_cross_ln_gamma'] = np.ones(embed_dim)
        params[f'dec_{l}_cross_ln_beta'] = np.zeros(embed_dim)

        # FFN
        params[f'dec_{l}_ff_w1'] = np.random.randn(embed_dim, embed_dim*4) * 0.02
        params[f'dec_{l}_ff_b1'] = np.zeros(embed_dim*4)
        params[f'dec_{l}_ff_w2'] = np.random.randn(embed_dim*4, embed_dim) * 0.02
        params[f'dec_{l}_ff_b2'] = np.zeros(embed_dim)
        params[f'dec_{l}_ff_ln_gamma'] = np.ones(embed_dim)
        params[f'dec_{l}_ff_ln_beta'] = np.zeros(embed_dim)
    # 交叉注意力投影（共享所有层）
    params['attn_w_q'] = np.random.randn(embed_dim, embed_dim) * 0.02
    params['attn_b_q'] = np.zeros(embed_dim)
    params['attn_w_k'] = np.random.randn(embed_dim, embed_dim) * 0.02
    params['attn_b_k'] = np.zeros(embed_dim)
    params['attn_w_v'] = np.random.randn(embed_dim, embed_dim) * 0.02   # 如果不使用 V 投影，可以忽略，但 location_sensitive_attention 内部可能用了 V，但它的参数传递是 keys 和 values 分离，但投影通常只对 query 和 key 做，value 不投影。在 location_sensitive_attention 中，我们只对 query 和 key 做了投影，没有对 value 做投影。但我们需要 attn_w_loc 和 attn_b_loc。
    params['attn_w_loc'] = np.random.randn(1, embed_dim) * 0.02
    params['attn_b_loc'] = np.zeros(embed_dim)
    return params


# ============================================================
# 8. 训练循环
# ============================================================
def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)

def d_mse_loss(pred, target):
    return 2.0 * (pred - target) / pred.size
def cosine_decay(epoch, total_epochs, lr_init=1e-4, lr_min=1e-6):
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(epoch / total_epochs * np.pi))
def warmup_cosine(epoch, warmup_epochs=10, total_epochs=300, lr_init=1e-4, lr_min=1e-6):
    if epoch < warmup_epochs:
        return lr_init * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(progress * np.pi))
def train():
    # 数据
#    sample = create_sample()
    sample = load_sample_from_dataset()
    text_ids = sample['text_ids']
#mel_target = sample['mel'][:,1:10,:]
    mel_target = sample['mel']
    vocab_size = sample['vocab_size']

    # 初始化参数
    best_loss = float('inf')
    params = init_tts_params(vocab_size, embed_dim=256, num_heads=8, num_encoder_layers=3, mel_dim=80)
    # 3. 尝试加载之前保存的最佳模型
    if os.path.exists('tts_best.npz'):
        data = np.load('tts_best.npz', allow_pickle=True)
        best_params = {key: data[key].item() if data[key].dtype == np.dtype('O') else data[key] for key in data.files}
        best_loss = float(data['best_loss'])
        # 只更新模型权重，不改变结构参数
        for key in best_params:
            if key in params:
                params[key] = best_params[key]
        print("✅ 从 tts_best.npz 恢复模型参数")
    else:
        print("ℹ️ 未找到 tts_best.npz，从头开始训练")

    # 训练
    epochs =500 
    lr_init = 0.00001
    step = 1

    for epoch in range(epochs):
        lr = cosine_decay(epoch, total_epochs=epochs, lr_init=lr_init, lr_min=1e-8)
        # 前向
        pred_mel, caches = tts(params, text_ids, mel_target, teacher_forcing=True)
        target_mel = mel_target[:,1:,:] # 去掉第一帧
        loss = mse_loss(pred_mel, target_mel)

        # 反向
        dloss = d_mse_loss(pred_mel, target_mel)
        grads = d_tts(dloss, caches, params)
        for key in grads:
            grads[key] = np.clip(grads[key], -1.0, 1.0)
        # 更新
        params, step = adam_update(params, grads, lr, step)
		# ---- SGD 更新 ----
#        for key in params:
#            if key in grads:
#                params[key] -= lr * grads[key]
        # ---- 权重剪裁 ----
#        clip_weight = 5.0   # 限制权重在 [-1.0, 1.0] 之间，可按需调整
#        for key in params:
#            # 只裁剪权重矩阵，不裁剪偏置和LayerNorm参数（可选）
#            if key.endswith('_w') or key.endswith('_w_q') or key.endswith('_w_k') or key.endswith('_w_v') or key.endswith('_w_o') or key.endswith('_w_loc') or key.endswith('_w_out'):
#                params[key] = np.clip(params[key], -clip_weight, clip_weight)
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        for epoch in range(epochs):
        # ... 训练代码 ...
            if loss < best_loss:
                best_loss = loss
                np.savez('tts_best.npz', **params, step=step,best_loss=best_loss)
                print(f"✅ 最佳模型已保存，loss={best_loss:.6f}")

    print("训练完成！")
	# 训练完成后保存模型参数
    np.savez('tts_model.npz', **params)
    print("✅ 模型参数已保存到 tts_model.npz")
    return params

if __name__ == "__main__":
    train()
