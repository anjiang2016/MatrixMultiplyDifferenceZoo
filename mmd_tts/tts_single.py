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
    relu, d_relu,
    silu, d_silu,
    softmax, d_softmax,
    layer_norm, d_layer_norm,
    matmul,
    adam_update
)
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
def load_sample_from_dataset(npz_path='tts_data.npz', sample_idx=0):
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
# 4. Location-Sensitive Attention（前向 + 反向）
# ============================================================
def attention(params, query, keys, values, location_features):
    """query: (batch, embed_dim), keys: (batch, T, embed_dim), location_features: (batch, T)"""
    batch, T, embed_dim = keys.shape
    query_exp = np.tile(query[:, None, :], (1, T, 1))

    q_proj, q_cache = linear(query_exp, params['attn_w_q'], params['attn_b_q'])
    k_proj, k_cache = linear(keys, params['attn_w_k'], params['attn_b_k'])
    loc_proj, loc_cache = linear(location_features[:, :, None], params['attn_w_loc'], params['attn_b_loc'])
    loc_proj = loc_proj.squeeze(-1)

    scores = np.sum(q_proj * k_proj, axis=-1) + loc_proj
    scores = scores / np.sqrt(embed_dim)
    attn, softmax_cache = softmax(scores, axis=-1)
    context = np.sum(attn[:, :, None] * values, axis=1)

    cache = {
        'query': query, 'keys': keys, 'values': values,
        'location_features': location_features,
        'q_cache': q_cache, 'k_cache': k_cache, 'loc_cache': loc_cache,
		'q_proj':k_proj, 'k_proj':k_proj,
        'scores': scores, 'softmax_cache': softmax_cache,
        'attn': attn, 'context': context,
        'T': T, 'embed_dim': embed_dim,
    }
    return context, attn, cache

def d_attention(dcontext, dattn, cache):
    # dcontext: (batch, embed_dim), dattn: (batch, T)
    attn = cache['attn']
    values = cache['values']
    T = cache['T']
    embed_dim = cache['embed_dim']

    # context = sum(attn * values)
    dattn_from_context = np.sum(dcontext[:, None, :] * values, axis=-1)  # (batch, T)
    dattn_total = dattn + dattn_from_context

    dscores = d_softmax(dattn_total, cache['softmax_cache'])
    dscores = dscores / np.sqrt(embed_dim)

    #dq_proj, dk_proj, dloc_proj = dscores, dscores, dscores  # 因为 scores = q_proj*k_proj + loc_proj
    # 正确计算 q_proj, k_proj, loc_proj 的梯度
    dq_proj = dscores[:, :, None] * cache['k_proj']   # (batch, T, embed_dim)
    dk_proj = dscores[:, :, None] * cache['q_proj']   # (batch, T, embed_dim)
    dloc_proj = dscores[:, :, None]                   # (batch, T, 1)

    # 反向通过线性层
    dx_q, dw_q, db_q = d_linear(dq_proj, cache['q_cache'])
    dx_k, dw_k, db_k = d_linear(dk_proj, cache['k_cache'])
    dx_loc, dw_loc, db_loc = d_linear(dloc_proj[:, :, None], cache['loc_cache'])

    # 累加梯度到 keys 和 location_features
    dkeys = dx_k
    dlocation = dx_loc.squeeze(-1)

    # query 的梯度是 dx_q 在时间维度的累加
    dquery = dx_q.sum(axis=1)

    # dvalues: context 对 values 的梯度
    dvalues = np.sum(dcontext[:, None, :] * attn[:, :, None], axis=1)

    grads = {
        'attn_w_q': dw_q, 'attn_b_q': db_q,
        'attn_w_k': dw_k, 'attn_b_k': db_k,
        'attn_w_loc': dw_loc, 'attn_b_loc': db_loc,
    }
    return dquery, dkeys, dvalues, dlocation, grads
def location_sensitive_attention(params, query, keys, values, location_features,position_bias):
    """
    query: (batch, embed_dim) 解码器当前步的隐状态
    keys: (batch, T, embed_dim) 编码器输出
    values: (batch, T, embed_dim) 编码器输出
    location_features: (batch, T) 累积注意力权重（初始全零）
    返回: context (batch, embed_dim), attn (batch, T), 更新后的 location_features
    """
#    import pdb;pdb.set_trace()
    batch, T, embed_dim = keys.shape
    
    # 1. 对 query 做线性变换 (与 keys 维度匹配)
    q_proj = np.matmul(query, params['attn_w_q']) + params['attn_b_q']  # (batch, embed_dim)
    q_proj = np.tile(q_proj[:, None, :], (1, T, 1))  # (batch, T, embed_dim)
    
    # 2. 对 keys 做线性变换
    k_proj = np.matmul(keys, params['attn_w_k']) + params['attn_b_k']  # (batch, T, embed_dim)
    
    # 3. 对 location_features 做线性变换 (也可用卷积，但这里用线性简化)
    # location_features: (batch, T) -> (batch, T, 1) -> 线性映射 -> (batch, T, embed_dim)
    loc_proj = np.matmul(location_features[:, :, None], params['attn_w_loc']) + params['attn_b_loc']  # (batch, T, embed_dim)

    qk = np.sum(q_proj * k_proj, axis=-1)
    loc = np.sum(loc_proj, axis=-1)
#    print(f"qk std: {qk.std():.6f}, loc std: {loc.std():.6f}")
#    print(f"qk max: {qk.max():.6f}, loc max: {loc.max():.6f}")
    # 4. 计算分数 (逐元素相加)
    scores = np.sum(q_proj * k_proj, axis=-1) + np.sum(loc_proj, axis=-1)  # (batch, T)
    scores = scores / (np.sqrt(embed_dim))  # 缩放
#    scores = scores / 0.1  # 缩放
	# T = seq_len（文本长度，约 8）
    # t 是当前解码步数（0 ~ mel_len-1）
    scores = scores + position_bias
    
    # 5. softmax
    attn,softmax_cache = softmax(scores, axis=-1)  # (batch, T)
     
    # 6. 计算上下文向量
    context = np.sum(attn[:, :, None] * values, axis=1)  # (batch, embed_dim)
    
    # 7. 更新 location_features
#    location_features = location_features + attn
    
    return context, attn, location_features, {
        'query': query,
        'keys': keys,
        'values': values,
        'location_features': location_features,
        'attn': attn,
        'scores': scores,
        'q_proj': q_proj,
        'k_proj': k_proj,
        'loc_proj': loc_proj,
        'embed_dim': embed_dim,
        'T': T,
		'softmax_cache':softmax_cache,
    }
def d_location_sensitive_attention(dcontext, dattn_prev, cache,params):
    """
    dcontext: (batch, embed_dim) 上下文向量的梯度
    dattn_prev: (batch, T) 注意力权重的梯度（来自上层）
    cache: 包含前向的中间变量
    返回: (dquery, dkeys, dvalues, dlocation, grads)
    """
    # 从 cache 中提取
    query = cache['query']
    keys = cache['keys']
    values = cache['values']
    location_features = cache['location_features']
    attn = cache['attn']
    scores = cache['scores']
    q_proj = cache['q_proj']
    k_proj = cache['k_proj']
    loc_proj = cache['loc_proj']
    softmax_cache = cache['softmax_cache']
    embed_dim = cache['embed_dim']
    T = cache['T']
    batch = query.shape[0]
    
    # 合并 attn 的梯度
    dattn = dattn_prev + np.sum(dcontext[:, None, :] * values, axis=-1)      # (batch, T)
    
    # softmax 反向
    dscores = d_softmax(dattn, softmax_cache)                                # (batch, T)
    dscores = dscores / (np.sqrt(embed_dim))
#    dscores = dscores /0.1 
    
    # 各投影的梯度
    dq_proj = dscores[:, :, None] * k_proj                                  # (batch, T, embed_dim)
    dk_proj = dscores[:, :, None] * q_proj                                  # (batch, T, embed_dim)
    dloc_proj = dscores[:, :, None] * 1.0                                    # (batch, T, embed_dim)
    
    # query 的梯度
    dquery = np.sum(dq_proj, axis=1)                                         # (batch, embed_dim)
    # keys 的梯度
    dkeys = dk_proj                                                          # (batch, T, embed_dim)
    # values 的梯度
    dvalues = attn[:, :, None] * dcontext[:, None, :]                       # (batch, T, embed_dim)
    
    # ---- 修正 dlocation ----
    # loc_proj = location_features[:, :, None] @ attn_w_loc + b_loc
    # attn_w_loc 形状为 (1, embed_dim)
    # dlocation 应为 (batch, T)
#    dlocation = np.sum(dloc_proj * params['attn_w_loc'], axis=-1)            # (batch, T)
    dlocation = np.zeros_like(cache['location_features'])  # 前向不再更新，梯度为 0   
    # 参数梯度
    dw_q = np.matmul(query.T, dq_proj.sum(axis=1))                          # (embed_dim, embed_dim)
    db_q = dq_proj.sum(axis=(0,1))                                          # (embed_dim,)
    dw_k = np.matmul(keys.transpose(0,2,1), dk_proj).sum(axis=0)            # (embed_dim, embed_dim)
    db_k = dk_proj.sum(axis=(0,1))                                          # (embed_dim,)
    # attn_w_loc: (1, embed_dim)，梯度也为 (1, embed_dim)
    # 前向：loc_proj = location_features[:, :, None] @ attn_w_loc
    # 所以 dw_loc = (location_features[:, :, None].transpose(0,2,1) @ dloc_proj).sum(axis=0)
    # location_features[:, :, None].transpose(0,2,1) 形状 (batch, 1, T)
    # dloc_proj 形状 (batch, T, embed_dim)
    # 乘积 -> (batch, 1, embed_dim)，然后对 batch 求和
    dw_loc = np.matmul(location_features[:, :, None].transpose(0,2,1), dloc_proj).sum(axis=0)  # (1, embed_dim)
    db_loc = dloc_proj.sum(axis=(0,1))                                     # (embed_dim,)
    
    grads = {
        'attn_w_q': dw_q,
        'attn_b_q': db_q,
        'attn_w_k': dw_k,
        'attn_b_k': db_k,
        'attn_w_loc': dw_loc,
        'attn_b_loc': db_loc,
    }
    
    return dquery, dkeys, dvalues, dlocation, grads
'''
# ============================================================
# 5. 解码器（前向 + 反向）
# ============================================================
def decoder(params, encoder_output, mel_targets=None, teacher_forcing=True, max_len=50):
    batch, T, embed_dim = encoder_output.shape
    mel_dim = params['mel_dim']
    outputs = []
    attn_list = []

    prev_output = np.zeros((batch, mel_dim), dtype=np.float32)
    location_features = np.zeros((batch, T), dtype=np.float32)
    if teacher_forcing and mel_targets is not None:
        inputs = mel_targets[:, :-1, :]

        for t in range(inputs.shape[1]):

#if np.random.random() < teacher_forcing_ratio:
            
            prev_output = inputs[:, t, :]   # 使用真实值
            tmp = prev_output
#print(f"Step {t}: prev_output max = {tmp.max():.4f}, mean = {tmp.mean()}")
            query, q_cache = linear(prev_output, params['decoder_w_q'], params['decoder_b_q'])
			# 方式二：动态偏置（依赖 t，让注意力逐步移动）
            max_len = inputs.shape[1]
            target = (t / max_len) * T   # 期望的焦点位置
            #scores[:, target] += 2.0
            p_bias = -np.abs(np.arange(T) - target) * 0.5   # 离 target 越近，加分越多
            context, attn, location_features,attn_cache = location_sensitive_attention(params, query, encoder_output, encoder_output, location_features,p_bias)
			# ... 注意力计算 ...
            print(f"Step {t}: attn max = {attn.max():.4f}, argmax = {attn.argmax()}")
            # ...
#print(f"Step {t}: context max = {context.max():.4f}, mean = {context.mean()}")
            location_features = location_features + attn
            combined = np.concatenate([prev_output, context], axis=-1)
            mel_frame, mel_cache = linear(combined, params['decoder_w_out'], params['decoder_b_out'])
#mel_frame = mel_targets[:,t+1,:]
            prev_output = mel_frame
            outputs.append(mel_frame)
            attn_list.append(attn)
#            print(f"Step {t}: pre_output max = {prev_output.max():.4f}, mean = {prev_output.mean()}")
#            print(f"Step {t}: mel_frame max = {mel_frame.max():.4f}, mean = {mel_frame.mean()}")
        pred_mel = np.stack(outputs, axis=1)
    else:
        for tt in range(max_len):
            query, q_cache = linear(prev_output, params['decoder_w_q'], params['decoder_b_q'])
            context, attn,location_features, attn_cache = location_sensitive_attention(params, query, encoder_output, encoder_output, location_features)
			# ... 注意力计算 ...
            print(f"Step {tt}: attn max = {attn.max():.4f}, argmax = {attn.argmax()}")
            # ...
            location_features = location_features + attn
            combined = np.concatenate([prev_output, context], axis=-1)
            mel_frame, mel_cache = linear(combined, params['decoder_w_out'], params['decoder_b_out'])
            outputs.append(mel_frame)
            attn_list.append(attn)
            prev_output = mel_frame
#print(f"Step {tt}: pre_output max = {prev_output.max():.4f}, arg max = {prev_output.argmax()}")
        pred_mel = np.stack(outputs, axis=1)

    cache = {
        'encoder_output': encoder_output,
        'mel_targets': mel_targets,
        'prev_outputs': outputs,
        'attn_list': attn_list,
		'attn_cache': attn_cache,
        'location_features': location_features,
        'q_cache': q_cache, 'mel_cache': mel_cache,
        'T': T, 'mel_dim': mel_dim,
    }
    return pred_mel, cache

def d_decoder(dout_mel, cache, params):
    # dout_mel: (batch, mel_len, mel_dim)
    mel_targets = cache['mel_targets']
    outputs = cache['prev_outputs']
    attn_list = cache['attn_list']
    location_features = cache['location_features']
    encoder_output = cache['encoder_output']
    T = cache['T']

    grads = {}
    dx_encoder = np.zeros_like(encoder_output)
    dlocation = np.zeros_like(location_features)

    # 从后往前
    for t in range(len(outputs) - 1, -1, -1):
        dmel = dout_mel[:, t, :]
        dx_combined, dw_out, db_out = d_linear(dmel, cache['mel_cache'])
        grads['decoder_w_out'] = dw_out
        grads['decoder_b_out'] = db_out

        # 拆分 combined = concat(prev_output, context)
        mel_dim = params['mel_dim']
        dprev = dx_combined[:, :mel_dim]
        dcontext = dx_combined[:, mel_dim:]

        # 反向传播到上下文和注意力
        dquery, dkeys, dvalues, dloc, attn_grads = d_location_sensitive_attention(
            dcontext, np.zeros_like(attn_list[t]), cache['attn_cache'],params
        )
        # 累加梯度
        dx_encoder += dkeys + dvalues
        dlocation += dloc
        for k, v in attn_grads.items():
            grads[k] = v

    return grads, dx_encoder
'''
# ============================================================
# 3. Transformer 解码器层
# ============================================================
def transformer_decoder_layer(params, x, encoder_output, location_features, mask, layer_idx):
    """
    x: (batch, seq_len, embed_dim) 当前解码器状态
    encoder_output: (batch, T_enc, embed_dim) 编码器输出
    location_features: (batch, T_enc) 位置累积特征
    mask: (seq_len, seq_len) 因果掩码
    layer_idx: 当前层索引
    """
    batch, seq_len, embed_dim = x.shape
    T_enc = encoder_output.shape[1]
    num_heads = params['num_heads']
    head_dim = embed_dim // num_heads

    # ---- 1. 自注意力（因果掩码） ----
    # Q, K, V
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

    # 注意力分数 + 因果掩码
    scores = np.matmul(Q_rot, K_rot.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    scores = scores + mask[None, None, :, :]  # mask 形状 (1, 1, seq_len, seq_len)
    attn, softmax_cache = softmax(scores, axis=-1)
    attn_out = np.matmul(attn, V)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
    attn_out, attn_linear_cache = linear(attn_out, params[f'dec_{layer_idx}_self_w_o'], params[f'dec_{layer_idx}_self_b_o'])

    # 残差 + LayerNorm
    x = x + attn_out
    x, ln1_cache = layer_norm(x, params[f'dec_{layer_idx}_self_ln_gamma'], params[f'dec_{layer_idx}_self_ln_beta'])

    # ---- 2. 交叉注意力（使用 location_sensitive_attention） ----
    query, q_cross_cache = linear(x, params[f'dec_{layer_idx}_cross_w_q'], params[f'dec_{layer_idx}_cross_b_q'])
    # 调用你的 location_sensitive_attention
    context, attn_cross, location_features, attn_cache = location_sensitive_attention(
        params, query, encoder_output, encoder_output, location_features
    )
    # context: (batch, embed_dim)，需要扩展到每个位置
    context = np.tile(context[:, None, :], (1, seq_len, 1))  # (batch, seq_len, embed_dim)
    x = x + context
    x, ln2_cache = layer_norm(x, params[f'dec_{layer_idx}_cross_ln_gamma'], params[f'dec_{layer_idx}_cross_ln_beta'])

    # ---- 3. 前馈网络 ----
    ff_h, ff1_cache = linear(x, params[f'dec_{layer_idx}_ff_w1'], params[f'dec_{layer_idx}_ff_b1'])
    ff_h = relu(ff_h)
    ff_out, ff2_cache = linear(ff_h, params[f'dec_{layer_idx}_ff_w2'], params[f'dec_{layer_idx}_ff_b2'])
    x = x + ff_out
    x, ln3_cache = layer_norm(x, params[f'dec_{layer_idx}_ff_ln_gamma'], params[f'dec_{layer_idx}_ff_ln_beta'])

    cache = {
        'self_q_cache': q_cache, 'self_k_cache': k_cache, 'self_v_cache': v_cache,
        'self_softmax_cache': softmax_cache,
        'self_attn_linear_cache': attn_linear_cache,
        'self_ln1_cache': ln1_cache,
        'cross_q_cache': q_cross_cache,
        'cross_attn_cache': attn_cache,
        'cross_ln_cache': ln2_cache,
        'ff1_cache': ff1_cache, 'ff2_cache': ff2_cache,
        'ff_ln_cache': ln3_cache,
        'attn_cross': attn_cross,
        'location_features': location_features,
    }
    return x, location_features, cache

# ============================================================
# 4. 完整的 Transformer 解码器
# ============================================================
def decoder(params, encoder_output, mel_targets=None, teacher_forcing=True, max_len=200):
    """
    params: 模型参数
    encoder_output: (batch, T_enc, embed_dim)
    mel_targets: (batch, mel_len, mel_dim) 训练时提供
    teacher_forcing: 是否使用真实目标作为输入
    max_len: 最大生成帧数
    """
    batch, T_enc, embed_dim = encoder_output.shape
    mel_dim = params['mel_dim']
    num_layers = params['decoder_layers']
    dropout_rate = params.get('dropout_rate', 0.1)

    # ---- 初始化 ----
    prev_output = np.zeros((batch, mel_dim), dtype=np.float32)
    outputs = []
    attn_list = []
    location_features = np.zeros((batch, T_enc), dtype=np.float32)

    # 因果掩码（固定）
    mask = causal_mask(max_len)

    # 解码器位置编码（RoPE）
    cos_emb = params['decoder_cos_emb'][:max_len, :]
    sin_emb = params['decoder_sin_emb'][:max_len, :]

    # ---- 自回归循环 ----
    for t in range(max_len):
        # 1. Pre-Net（降维 + Dropout）
        if teacher_forcing and mel_targets is not None:
            prev_output = mel_targets[:, t, :] if t > 0 else np.zeros((batch, mel_dim))
        pre_out, pre_cache = linear(prev_output, params['decoder_pre_w'], params['decoder_pre_b'])
        pre_out = dropout(pre_out, dropout_rate)

        # 2. 添加位置编码
        x = pre_out + np.concatenate([cos_emb[t:t+1, :], sin_emb[t:t+1, :]], axis=1)[:, :embed_dim]  # 简化，直接用 RoPE

        # 3. Transformer 层
        layer_caches = {}
        for l in range(num_layers):
            x, location_features, layer_cache = transformer_decoder_layer(
                params, x, encoder_output, location_features, mask[:t+1, :t+1], l
            )
            layer_caches[f'layer_{l}'] = layer_cache

        # 4. 输出 Mel 帧
        mel_frame, mel_cache = linear(x[:, -1, :], params['decoder_w_out'], params['decoder_b_out'])
        outputs.append(mel_frame)
        attn_list.append(layer_caches['layer_0']['attn_cross'])

        # 5. 更新 prev_output（自回归）
        if not teacher_forcing:
            prev_output = mel_frame

        # 如果达到目标长度或生成结束，停止
        if t >= max_len - 1:
            break

    pred_mel = np.stack(outputs, axis=1)  # (batch, mel_len, mel_dim)

    cache = {
        'outputs': outputs,
        'attn_list': attn_list,
        'layer_caches': layer_caches,
        'mel_cache': mel_cache,
        'pre_cache': pre_cache,
        'location_features': location_features,
        'mask': mask,
    }
    return pred_mel, cache
def d_decoder(dout_mel, cache, params):
    """
    反向传播：从输出到解码器输入和编码器输出
    dout_mel: (batch, mel_len, mel_dim) 损失对输出 mel 的梯度
    cache: decoder 前向传播中保存的缓存
    params: 模型参数
    返回: (grads, dx_encoder)
        grads: 参数字典
        dx_encoder: (batch, T_enc, embed_dim) 编码器输出的梯度
    """
    grads = {}
    outputs = cache['outputs']  # list of (batch, mel_dim)
    attn_list = cache['attn_list']
    layer_caches = cache['layer_caches']
    mel_cache = cache.get('mel_cache')
    pre_cache = cache.get('pre_cache')
    location_features = cache['location_features']
    mask = cache['mask']
    mel_len = dout_mel.shape[1]
    batch, T_enc, embed_dim = dout_mel.shape[0], cache['layer_caches']['layer_0']['attn_cross'].shape[1], params['embed_dim']
    
    num_layers = params['decoder_layers']
    dropout_rate = params.get('dropout_rate', 0.1)

    # ---- 1. 初始化从输出层回传的梯度 ----
    dx = dout_mel  # (batch, mel_len, mel_dim)
    
    # ---- 2. 逐层反向传播 ----
    for l in range(num_layers - 1, -1, -1):
        layer = layer_caches[f'layer_{l}']
        
        # 提取当前层的缓存
        self_q_cache = layer['self_q_cache']
        self_k_cache = layer['self_k_cache']
        self_v_cache = layer['self_v_cache']
        self_softmax_cache = layer['self_softmax_cache']
        self_attn_linear_cache = layer['self_attn_linear_cache']
        self_ln1_cache = layer['self_ln1_cache']
        cross_q_cache = layer['cross_q_cache']
        cross_attn_cache = layer['cross_attn_cache']
        cross_ln_cache = layer['cross_ln_cache']
        ff1_cache = layer['ff1_cache']
        ff2_cache = layer['ff2_cache']
        ff_ln_cache = layer['ff_ln_cache']
        attn_cross = layer['attn_cross']
        location_features = layer['location_features']
        
        # ---- 2a. FFN 反向 ----
        # 从当前层的输出开始
        dx_ff, dgamma_ff, dbeta_ff = d_layer_norm(dx, ff_ln_cache)
        grads[f'dec_{l}_ff_ln_gamma'] = dgamma_ff
        grads[f'dec_{l}_ff_ln_beta'] = dbeta_ff
        
        # FFN 第二层线性
        dx_ff, dw_ff2, db_ff2 = d_linear(dx_ff, ff2_cache)
        grads[f'dec_{l}_ff_w2'] = dw_ff2
        grads[f'dec_{l}_ff_b2'] = db_ff2
        
        # ReLU 反向
        dx_ff = d_relu(dx_ff, ff2_cache['x'])  # ff2_cache['x'] 是 ReLU 的输入
        
        # FFN 第一层线性
        dx_ff, dw_ff1, db_ff1 = d_linear(dx_ff, ff1_cache)
        grads[f'dec_{l}_ff_w1'] = dw_ff1
        grads[f'dec_{l}_ff_b1'] = db_ff1
        
        # 残差：x = x + ff_out，累加 dx_ff 到 dx
        dx = dx_ff + dx
        
        # ---- 2b. 交叉注意力反向 ----
        # 首先通过 LayerNorm
        dx_cross, dgamma_cross, dbeta_cross = d_layer_norm(dx, cross_ln_cache)
        grads[f'dec_{l}_cross_ln_gamma'] = dgamma_cross
        grads[f'dec_{l}_cross_ln_beta'] = dbeta_cross
        
        # 交叉注意力的输入是 x，经过 linear 得到 query
        # 但我们需要把梯度传递回 x，以及编码器输出的梯度
        
        # 注意：交叉注意力将 context 扩展到了每个位置，然后 x = x + context
        # 所以 dx_cross 的梯度需要拆分：一部分用于 context，一部分用于 x
        
        # 获取交叉注意力的 query 的梯度
        dquery, dw_q_cross, db_q_cross = d_linear(dx_cross, cross_q_cache)
        grads[f'dec_{l}_cross_w_q'] = dw_q_cross
        grads[f'dec_{l}_cross_b_q'] = db_q_cross
        
        # 调用 d_location_sensitive_attention 计算交叉注意力的梯度
        # 注意：cross_attn_cache 包含了 attention 前向的所有信息
        dquery_attn, dkeys_attn, dvalues_attn, dlocation_attn, attn_grads = d_location_sensitive_attention(
            dquery,  # 对 context 的梯度（因为 context = attention 输出）
            np.zeros_like(attn_cross),  # 对 attn 的梯度（我们不需要）
            cross_attn_cache,
            params
        )
        
        # 累加注意力参数梯度
        for k, v in attn_grads.items():
            if k not in grads:
                grads[k] = np.zeros_like(v)
            grads[k] += v
        
        # 编码器输出的梯度来自 dkeys_attn 和 dvalues_attn
        dx_encoder_from_cross = dkeys_attn + dvalues_attn  # 因为 keys = values = encoder_output
        
        # 残差：x = x + context，所以 dx 需要加上 attention 的梯度
        dx = dquery_attn + dx  # 注意：dquery_attn 是 query 的梯度，也就是 x 的梯度
        
        # ---- 2c. 自注意力反向 ----
        # 通过 LayerNorm
        dx_self, dgamma_self, dbeta_self = d_layer_norm(dx, self_ln1_cache)
        grads[f'dec_{l}_self_ln_gamma'] = dgamma_self
        grads[f'dec_{l}_self_ln_beta'] = dbeta_self
        
        # 自注意力的输出线性层反向
        dx_self_attn, dw_o_self, db_o_self = d_linear(dx_self, self_attn_linear_cache)
        grads[f'dec_{l}_self_w_o'] = dw_o_self
        grads[f'dec_{l}_self_b_o'] = db_o_self
        
        # 恢复形状 (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        batch, seq_len, embed_dim = dx_self_attn.shape
        num_heads = params['num_heads']
        head_dim = embed_dim // num_heads
        dx_self_attn = dx_self_attn.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        
        # 自注意力的 V 和 attn 的梯度
        # 从缓存中获取 V 和 attn
        V = layer['self_v_cache']['x']  # 原始 V 的值（需要从缓存中获取）
        V = V.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        
        # 因为 attn_out = attn @ V，所以：
        dattn_self = np.matmul(dx_self_attn, V.transpose(0, 1, 3, 2))  # dout @ V^T
        dV_self = np.matmul(layer['self_softmax_cache']['x'].transpose(0, 1, 3, 2), dx_self_attn)  # attn^T @ dout
        
        # softmax 反向
        dscores_self, _ = d_softmax(dattn_self, layer['self_softmax_cache'])
        dscores_self = dscores_self / np.sqrt(head_dim)
        
        # 应用因果掩码的梯度（mask 无参数，直接忽略）
        # 但需要将 mask 区域的梯度置零
        mask_matrix = mask[:seq_len, :seq_len]
        dscores_self = dscores_self * (mask_matrix == 0)  # 只保留未被屏蔽的位置
        
        # K_rot 和 Q_rot 的梯度
        Q_rot = layer['self_Q_rot']  # 从缓存中获取
        K_rot = layer['self_K_rot']
        dK_rot_self = np.matmul(dscores_self.transpose(0, 1, 3, 2), Q_rot)
        dQ_rot_self = np.matmul(dscores_self, K_rot)
        
        # RoPE 反向
        cos_emb = params['decoder_cos_emb'][:seq_len, :]
        sin_emb = params['decoder_sin_emb'][:seq_len, :]
        dQ_self, dK_self = d_apply_rotary(dQ_rot_self, dK_rot_self, 
                                           layer['self_Q'], layer['self_K'], 
                                           cos_emb, sin_emb)
        
        # 重塑回 (batch, seq_len, embed_dim)
        dQ_self = dQ_self.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        dK_self = dK_self.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        dV_self = dV_self.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        
        # Q, K, V 的线性层反向
        dx_self_q, dw_q_self, db_q_self = d_linear(dQ_self, self_q_cache)
        dx_self_k, dw_k_self, db_k_self = d_linear(dK_self, self_k_cache)
        dx_self_v, dw_v_self, db_v_self = d_linear(dV_self, self_v_cache)
        
        grads[f'dec_{l}_self_w_q'] = dw_q_self
        grads[f'dec_{l}_self_b_q'] = db_q_self
        grads[f'dec_{l}_self_w_k'] = dw_k_self
        grads[f'dec_{l}_self_b_k'] = db_k_self
        grads[f'dec_{l}_self_w_v'] = dw_v_self
        grads[f'dec_{l}_self_b_v'] = db_v_self
        
        # 累加梯度
        dx = dx_self_q + dx_self_k + dx_self_v + dx_self_q  # 残差：x = x + attn_out
        
        # 将编码器输出的梯度从这一层传递出去
        # 注意：这里需要将 dx_encoder_from_cross 累加到总梯度上
        # 但 dx_encoder_from_cross 是在交叉注意力中计算的，已经在循环中累加了
        # 但我们还没有把它存下来，需要在循环中累加
        dx_encoder = dx_encoder_from_cross if l == num_layers - 1 else dx_encoder + dx_encoder_from_cross
    
    # ---- 3. Pre-Net 和输出层反向 ----
    # Pre-Net 反向：从最后一层的输入 dx 开始
    # dx 是最后一层输入（经过 pre-net 后的梯度）
    dx_pre, dw_pre, db_pre = d_linear(dx, pre_cache)  # 需要 pre_cache 保存
    grads['decoder_pre_w'] = dw_pre
    grads['decoder_pre_b'] = db_pre
    
    # 输出层反向：从 mel 输出的梯度开始
    dx_mel, dw_out, db_out = d_linear(dout_mel, mel_cache)
    grads['decoder_w_out'] = dw_out
    grads['decoder_b_out'] = db_out
    
    # ---- 4. 返回 ----
    return grads, dx_encoder

# ============================================================
# 6. 完整 TTS 模型（前向 + 反向）
# ============================================================
def tts(params, text_ids, mel_targets=None, teacher_forcing=True):
    encoder_output, enc_caches = encoder(params, text_ids)
    pred_mel, dec_caches = decoder(params, encoder_output, mel_targets, teacher_forcing)
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
def init_tts_params(vocab_size, embed_dim=128, num_heads=8, num_encoder_layers=3, mel_dim=80, max_seq_len=100):
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
    '''
    # 解码器
    params['decoder_w_q'] = np.random.randn(mel_dim, embed_dim) * 0.2
    params['decoder_b_q'] = np.zeros(embed_dim)
    params['decoder_w_out'] = np.random.randn(embed_dim + mel_dim, mel_dim) * 0.2
    params['decoder_b_out'] = np.zeros(mel_dim)

    # 注意力
    params['attn_w_q'] = np.random.randn(embed_dim, embed_dim) * 0.1
    params['attn_b_q'] = np.zeros(embed_dim)
    params['attn_w_k'] = np.random.randn(embed_dim, embed_dim) * 0.1
    params['attn_b_k'] = np.zeros(embed_dim)
#   params['attn_w_loc'] = np.random.randn(1, 1) * 0.02
#   params['attn_b_loc'] = np.zeros(1)
    params['attn_w_loc'] = np.random.randn(1, embed_dim) * 0.1  # 或者 (1, embed_dim)
    params['attn_b_loc'] = np.zeros(embed_dim)
    params['decoder_layers'] = num_layers
    params['num_heads'] = num_heads
    params['mel_dim'] = mel_dim
    params['dropout_rate'] = dropout_rate
    '''
    # 位置编码
    head_dim = embed_dim // num_heads
    cos_emb, sin_emb = get_rotary_embedding(max_len, head_dim)
    params['decoder_cos_emb'] = cos_emb
    params['decoder_sin_emb'] = sin_emb

    # Pre-Net
    params['decoder_pre_w'] = np.random.randn(mel_dim, embed_dim) * 0.02
    params['decoder_pre_b'] = np.zeros(embed_dim)

    # 输出层
    params['decoder_w_out'] = np.random.randn(embed_dim, mel_dim) * 0.02
    params['decoder_b_out'] = np.zeros(mel_dim)

    # 各层
    for l in range(num_layers):
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
        # 注意：attn_w_k, attn_w_v, attn_w_o, attn_b_k, attn_b_v, attn_b_o 已在 location_sensitive_attention 中定义
        # 这里只需要 cross_w_q 和 cross_b_q
        params[f'dec_{l}_cross_ln_gamma'] = np.ones(embed_dim)
        params[f'dec_{l}_cross_ln_beta'] = np.zeros(embed_dim)

        # FFN
        params[f'dec_{l}_ff_w1'] = np.random.randn(embed_dim, embed_dim*4) * 0.02
        params[f'dec_{l}_ff_b1'] = np.zeros(embed_dim*4)
        params[f'dec_{l}_ff_w2'] = np.random.randn(embed_dim*4, embed_dim) * 0.02
        params[f'dec_{l}_ff_b2'] = np.zeros(embed_dim)
        params[f'dec_{l}_ff_ln_gamma'] = np.ones(embed_dim)
        params[f'dec_{l}_ff_ln_beta'] = np.zeros(embed_dim)
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
    epochs =3000 
    lr_init = 0.0001
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
#        params, step = adam_update(params, grads, lr, step)
		# ---- SGD 更新 ----
        for key in params:
            if key in grads:
                params[key] -= lr * grads[key]
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
