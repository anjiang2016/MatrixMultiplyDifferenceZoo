"""
model.py
--------
Transformer 解码器（纯函数式）
每个前向函数返回 (out, cache)，每个反向函数接收 (dout, cache, params) 返回 (dx, grads)
"""

import numpy as np
from funcs import (
    linear, d_linear,
    relu, d_relu,
    softmax, d_softmax,
    dropout, d_dropout,
    layer_norm, d_layer_norm
)

# ========== 1. 旋转位置编码 ==========
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
    """q,k: (batch, heads, seq, head_dim)"""
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
    q1, q2 = q[..., :d], q[..., d:]
    k1, k2 = k[..., :d], k[..., d:]
    cos = cos_emb[None, None, :, :d]
    sin = sin_emb[None, None, :, :d]
    
    dq1_rot, dq2_rot = dq_rot[..., :d], dq_rot[..., d:]
    dk1_rot, dk2_rot = dk_rot[..., :d], dk_rot[..., d:]
    
    dq1 = dq1_rot * cos + dq2_rot * sin
    dq2 = -dq1_rot * sin + dq2_rot * cos
    dk1 = dk1_rot * cos + dk2_rot * sin
    dk2 = -dk1_rot * sin + dk2_rot * cos
    
    dq = np.concatenate([dq1, dq2], axis=-1)
    dk = np.concatenate([dk1, dk2], axis=-1)
    return dq, dk

# ========== 2. 多头注意力 ==========
def multi_head_attention(params, x, mask=None):
    batch, seq_len, embed_dim = x.shape
    num_heads = params['num_heads']
    head_dim = embed_dim // num_heads

    # Q, K, V 投影
    Q, _ = linear(x, params['w_q'], params['b_q'])
    K, _ = linear(x, params['w_k'], params['b_k'])
    V, _ = linear(x, params['w_v'], params['b_v'])

    # 重塑为多头格式
    Q = Q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # RoPE
    cos_emb = params['cos_emb'][:seq_len, :]
    sin_emb = params['sin_emb'][:seq_len, :]
    Q_rot, K_rot = apply_rotary(Q, K, cos_emb, sin_emb)

    # 注意力分数
    scores = np.matmul(Q_rot, K_rot.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask * -1e9
    
    attn, _ = softmax(scores, axis=-1)
    attn, attn_mask = dropout(attn, params.get('dropout_rate', 0.0))
    
    out = np.matmul(attn, V)  # (batch, heads, seq, head_dim)
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
    out, _ = linear(out, params['w_o'], params['b_o'])

    cache = {
        'x': x,
        'Q': Q, 'K': K, 'V': V,
        'Q_rot': Q_rot, 'K_rot': K_rot,
        'scores': scores,
        'attn': attn,
        'attn_mask': attn_mask,
        'cos_emb': cos_emb, 'sin_emb': sin_emb,
        'num_heads': num_heads, 'head_dim': head_dim,
        'embed_dim': embed_dim,
        'mask': mask,
        'dropout_rate': params.get('dropout_rate', 0.0)
    }
    return out, cache

def d_multi_head_attention(dout, cache, params):
    x = cache['x']
    Q = cache['Q']; K = cache['K']; V = cache['V']
    Q_rot = cache['Q_rot']; K_rot = cache['K_rot']
    scores = cache['scores']
    attn = cache['attn']
    attn_mask = cache['attn_mask']
    cos_emb = cache['cos_emb']; sin_emb = cache['sin_emb']
    num_heads = cache['num_heads']; head_dim = cache['head_dim']
    embed_dim = cache['embed_dim']
    batch, seq_len, _ = x.shape

    # 输出线性层反向
    dout_reshape, dw_o, db_o = d_linear(dout, cache['out_reshape'] if 'out_reshape' in cache else x, params['w_o'], params['b_o'])
    grads = {'w_o': dw_o, 'b_o': db_o}

    # 恢复形状
    dout_attn = dout_reshape.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # V 的梯度
    dattn, dV = d_matmul(dout_attn, attn, V)  # 需要实现 d_matmul 或直接用 np.matmul 推导
    
    # 如果 d_matmul 未实现，手动计算：
    # dV = np.matmul(attn.transpose(0,1,3,2), dout_attn)
    # dattn = np.matmul(dout_attn, V.transpose(0,1,3,2))
    
    # dropout 反向
    dattn, _ = d_dropout(dattn, attn_mask)
    
    # softmax 反向
    dscores, _ = d_softmax(dattn, scores)
    dscores = dscores / np.sqrt(head_dim)

    # K_rot 和 Q_rot 的梯度
    dK_rot = np.matmul(dscores.transpose(0, 1, 3, 2), Q_rot)
    dQ_rot = np.matmul(dscores, K_rot)

    # RoPE 反向
    dQ, dK = d_apply_rotary(dQ_rot, dK_rot, Q, K, cos_emb, sin_emb)

    # 重塑回 (batch, seq, embed)
    dQ_reshape = dQ.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
    dK_reshape = dK.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
    dV_reshape = dV.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)

    # 三个线性层反向
    dx_q, dw_q, db_q = d_linear(dQ_reshape, x, params['w_q'], params['b_q'])
    dx_k, dw_k, db_k = d_linear(dK_reshape, x, params['w_k'], params['b_k'])
    dx_v, dw_v, db_v = d_linear(dV_reshape, x, params['w_v'], params['b_v'])

    dx = dx_q + dx_k + dx_v
    grads.update({'w_q': dw_q, 'b_q': db_q, 'w_k': dw_k, 'b_k': db_k, 'w_v': dw_v, 'b_v': db_v})
    return dx, grads

# ========== 3. 前馈网络 ==========
def feed_forward(params, x):
    h, _ = linear(x, params['ff_w1'], params['ff_b1'])
    h = relu(h)
    h, h_mask = dropout(h, params.get('dropout_rate', 0.0))
    out, _ = linear(h, params['ff_w2'], params['ff_b2'])
    cache = {'x': x, 'h': h, 'h_mask': h_mask, 'dropout_rate': params.get('dropout_rate', 0.0)}
    return out, cache

def d_feed_forward(dout, cache, params):
    x = cache['x']
    h = cache['h']
    h_mask = cache['h_mask']
    
    dh, dw2, db2 = d_linear(dout, h, params['ff_w2'], params['ff_b2'])
    dh, _ = d_dropout(dh, h_mask)
    dh = d_relu(dh, h)  # h 是 relu 后的值，但 d_relu 需要 relu 前的输入，这里 h 作为 x 可近似
    
    dx, dw1, db1 = d_linear(dh, x, params['ff_w1'], params['ff_b1'])
    grads = {'ff_w1': dw1, 'ff_b1': db1, 'ff_w2': dw2, 'ff_b2': db2}
    return dx, grads

# ========== 4. Transformer 层 ==========
def transformer_layer(params, x, mask=None):
    # 自注意力
    attn_out, attn_cache = multi_head_attention(params, x, mask)
    x = x + attn_out
    x, ln1_cache = layer_norm(x, params['ln1_gamma'], params['ln1_beta'])
    
    # 前馈
    ff_out, ff_cache = feed_forward(params, x)
    x = x + ff_out
    x, ln2_cache = layer_norm(x, params['ln2_gamma'], params['ln2_beta'])
    
    cache = {
        'attn_cache': attn_cache,
        'ff_cache': ff_cache,
        'ln1_cache': ln1_cache,
        'ln2_cache': ln2_cache,
        'x_before_attn': x,  # 用于残差
        'ln1_gamma': params['ln1_gamma'], 'ln1_beta': params['ln1_beta'],
        'ln2_gamma': params['ln2_gamma'], 'ln2_beta': params['ln2_beta']
    }
    return x, cache

def d_transformer_layer(dout, cache, params):
    # 反向顺序：ln2 -> ff -> ln1 -> attn
    dx, dgamma2, dbeta2 = d_layer_norm(dout, cache['ln2_cache'])
    dx_ff, grads_ff = d_feed_forward(dx, cache['ff_cache'], params)
    dx_ln1 = dx_ff + dx  # 残差
    
    dx, dgamma1, dbeta1 = d_layer_norm(dx_ln1, cache['ln1_cache'])
    dx_attn, grads_attn = d_multi_head_attention(dx, cache['attn_cache'], params)
    dx = dx_attn + dx  # 残差
    
    grads = {}
    grads.update(grads_ff)
    grads.update(grads_attn)
    grads['ln1_gamma'] = dgamma1
    grads['ln1_beta'] = dbeta1
    grads['ln2_gamma'] = dgamma2
    grads['ln2_beta'] = dbeta2
    return dx, grads

# ========== 5. 完整解码器 ==========
def transformer_decoder(params, input_ids, mask=None):
    batch, seq_len = input_ids.shape
    vocab_size = params['vocab_size']
    embed_dim = params['embed_dim']
    
    # 词嵌入
    one_hot = np.zeros((batch, seq_len, vocab_size), dtype=np.float32)
    one_hot[np.arange(batch)[:, None], np.arange(seq_len)[None, :], input_ids] = 1.0
    x, _ = linear(one_hot, params['embed_weight'], np.zeros(embed_dim))
    
    layer_caches = []
    for i in range(params['num_layers']):
        layer_params = {k: params[f'layer_{i}_{k}'] for k in [
            'w_q','b_q','w_k','b_k','w_v','b_v','w_o','b_o',
            'ln1_gamma','ln1_beta','ff_w1','ff_b1','ff_w2','ff_b2','ln2_gamma','ln2_beta'
        ]}
        layer_params['num_heads'] = params['num_heads']
        layer_params['cos_emb'] = params['cos_emb']
        layer_params['sin_emb'] = params['sin_emb']
        layer_params['dropout_rate'] = params.get('dropout_rate', 0.0)
        x, layer_cache = transformer_layer(layer_params, x, mask)
        layer_caches.append(layer_cache)
    
    logits, _ = linear(x, params['output_w'], params['output_b'])
    
    cache = {
        'input_ids': input_ids,
        'one_hot': one_hot,
        'x_embed': x,
        'layer_caches': layer_caches,
        'embed_weight': params['embed_weight'],
        'output_w': params['output_w'],
        'output_b': params['output_b']
    }
    return logits, cache

def d_transformer_decoder(dlogits, cache, params):
    # 输出线性层反向
    dx, dw_out, db_out = d_linear(dlogits, cache['x_embed'], params['output_w'], params['output_b'])
    grads = {'output_w': dw_out, 'output_b': db_out}
    
    # 逐层反向
    dx = dx
    layer_caches = cache['layer_caches']
    for i in range(params['num_layers'] - 1, -1, -1):
        layer_params = {k: params[f'layer_{i}_{k}'] for k in [
            'w_q','b_q','w_k','b_k','w_v','b_v','w_o','b_o',
            'ln1_gamma','ln1_beta','ff_w1','ff_b1','ff_w2','ff_b2','ln2_gamma','ln2_beta'
        ]}
        layer_params['num_heads'] = params['num_heads']
        layer_params['cos_emb'] = params['cos_emb']
        layer_params['sin_emb'] = params['sin_emb']
        layer_params['dropout_rate'] = params.get('dropout_rate', 0.0)
        dx, layer_grads = d_transformer_layer(dx, layer_caches[i], layer_params)
        for k, v in layer_grads.items():
            grads[f'layer_{i}_{k}'] = v
    
    # 词嵌入反向
    d_embed_weight = np.matmul(cache['one_hot'].transpose(0, 2, 1), dx).sum(axis=0)
    grads['embed_weight'] = d_embed_weight
    
    return None, grads

# ========== 6. 辅助函数 ==========
def d_matmul(dout, a, b):
    """dout 是 a @ b 的梯度"""
    da = np.matmul(dout, b.transpose(0, 1, 3, 2) if b.ndim == 4 else b.T)
    db = np.matmul(a.transpose(0, 1, 3, 2) if a.ndim == 4 else a.T, dout)
    return da, db

def init_model_params(vocab_size, embed_dim=256, num_layers=4, num_heads=8, max_seq_len=512, dropout_rate=0.1):
    head_dim = embed_dim // num_heads
    cos_emb, sin_emb = get_rotary_embedding(max_seq_len, head_dim)
    
    params = {
        'embed_dim': embed_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'vocab_size': vocab_size,
        'dropout_rate': dropout_rate,
        'embed_weight': np.random.randn(vocab_size, embed_dim) * 0.02,
        'output_w': np.random.randn(embed_dim, vocab_size) * 0.02,
        'output_b': np.zeros(vocab_size),
        'cos_emb': cos_emb,
        'sin_emb': sin_emb,
    }
    
    for i in range(num_layers):
        params[f'layer_{i}_w_q'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'layer_{i}_b_q'] = np.zeros(embed_dim)
        params[f'layer_{i}_w_k'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'layer_{i}_b_k'] = np.zeros(embed_dim)
        params[f'layer_{i}_w_v'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'layer_{i}_b_v'] = np.zeros(embed_dim)
        params[f'layer_{i}_w_o'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'layer_{i}_b_o'] = np.zeros(embed_dim)
        params[f'layer_{i}_ln1_gamma'] = np.ones(embed_dim)
        params[f'layer_{i}_ln1_beta'] = np.zeros(embed_dim)
        params[f'layer_{i}_ln2_gamma'] = np.ones(embed_dim)
        params[f'layer_{i}_ln2_beta'] = np.zeros(embed_dim)
        params[f'layer_{i}_ff_w1'] = np.random.randn(embed_dim, embed_dim*4) * 0.02
        params[f'layer_{i}_ff_b1'] = np.zeros(embed_dim*4)
        params[f'layer_{i}_ff_w2'] = np.random.randn(embed_dim*4, embed_dim) * 0.02
        params[f'layer_{i}_ff_b2'] = np.zeros(embed_dim)
    
    return params
