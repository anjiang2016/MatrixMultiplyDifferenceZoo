"""
model.py - Transformer 解码器（YOLO 风格）
所有线性层使用二维输入 (batch*seq, embed)，确保 funcs.d_linear 正常工作。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from funcs import (
    linear, d_linear,
    relu, d_relu,
    softmax, d_softmax,
    dropout, d_dropout,
    layer_norm, d_layer_norm
)

# ========== 辅助函数：RoPE ==========
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

# ========== 前向传播 ==========
def forward_transformer(params, input_ids, mask=None):
    batch, seq_len = input_ids.shape
    vocab_size = params['vocab_size']
    embed_dim = params['embed_dim']
    num_heads = params['num_heads']
    head_dim = embed_dim // num_heads
    num_layers = params['num_layers']
    dropout_rate = params.get('dropout_rate', 0.0)
    keep_prob = 1.0 - dropout_rate

    caches = {}

    # ---- 词嵌入 ----
    one_hot = np.zeros((batch, seq_len, vocab_size), dtype=np.float32)
    one_hot[np.arange(batch)[:, None], np.arange(seq_len)[None, :], input_ids] = 1.0
    # 展平为二维
    one_hot_flat = one_hot.reshape(-1, vocab_size)  # (batch*seq, vocab)
    x_flat, embed_cache = linear(one_hot_flat, params['embed_weight'], np.zeros(embed_dim))
    x = x_flat.reshape(batch, seq_len, embed_dim)   # (batch, seq, embed)
    caches['embed_cache'] = embed_cache
    caches['one_hot'] = one_hot

    # ---- 循环各层 ----
    for l in range(num_layers):
        layer_params = {k: params[f'layer_{l}_{k}'] for k in [
            'w_q','b_q','w_k','b_k','w_v','b_v','w_o','b_o',
            'ln1_gamma','ln1_beta','ff_w1','ff_b1','ff_w2','ff_b2','ln2_gamma','ln2_beta'
        ]}
        cos_emb = params['cos_emb'][:seq_len, :]
        sin_emb = params['sin_emb'][:seq_len, :]

        x_in = x.copy()
        x_flat = x.reshape(-1, embed_dim)  # (batch*seq, embed)

        # ---------- 多头自注意力 ----------
        Q_flat, q_cache = linear(x_flat, layer_params['w_q'], layer_params['b_q'])
        K_flat, k_cache = linear(x_flat, layer_params['w_k'], layer_params['b_k'])
        V_flat, v_cache = linear(x_flat, layer_params['w_v'], layer_params['b_v'])

        # 重塑为 (batch, seq, heads, head_dim)
        Q = Q_flat.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K_flat.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V_flat.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        Q_rot, K_rot = apply_rotary(Q, K, cos_emb, sin_emb)

        scores = np.matmul(Q_rot, K_rot.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask * -1e9

        attn, softmax_cache = softmax(scores, axis=-1)
        attn, dropout_cache = dropout(attn, keep_prob)

        attn_out = np.matmul(attn, V)  # (batch, heads, seq, head_dim)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        attn_out_flat = attn_out.reshape(-1, embed_dim)  # (batch*seq, embed)
        attn_out_flat, attn_linear_cache = linear(attn_out_flat, layer_params['w_o'], layer_params['b_o'])
        attn_out = attn_out_flat.reshape(batch, seq_len, embed_dim)

        x_res1 = x + attn_out
        x, ln1_cache = layer_norm(x_res1, layer_params['ln1_gamma'], layer_params['ln1_beta'])

        # ---------- 前馈网络 ----------
        x_flat = x.reshape(-1, embed_dim)
        ff_h_flat, ff_linear1_cache = linear(x_flat, layer_params['ff_w1'], layer_params['ff_b1'])
        ff_h = ff_h_flat.reshape(batch, seq_len, embed_dim*4)
        ff_h = relu(ff_h)
        ff_h_flat = ff_h.reshape(-1, embed_dim*4)
        ff_h_flat, ff_dropout_cache = dropout(ff_h_flat, keep_prob)
        ff_out_flat, ff_linear2_cache = linear(ff_h_flat, layer_params['ff_w2'], layer_params['ff_b2'])
        ff_out = ff_out_flat.reshape(batch, seq_len, embed_dim)

        x_res2 = x + ff_out
        x, ln2_cache = layer_norm(x_res2, layer_params['ln2_gamma'], layer_params['ln2_beta'])

        # 存储该层缓存（注意：所有 cache 中的 x 都是二维的，因为 linear 输入是二维的）
        caches[f'layer_{l}'] = {
            'x_in': x_in,                     # 三维输入，用于注意力形状恢复
            'x_flat': x_flat,                # 二维输入（用于 q/k/v 投影）
            'q_cache': q_cache,
            'k_cache': k_cache,
            'v_cache': v_cache,
            'Q': Q, 'K': K, 'V': V,
            'Q_rot': Q_rot, 'K_rot': K_rot,
            'scores': scores,
            'softmax_cache': softmax_cache,
            'attn': attn,
            'dropout_cache': dropout_cache,
            'attn_linear_cache': attn_linear_cache,
            'ln1_cache': ln1_cache,
            'ff_linear1_cache': ff_linear1_cache,
            'ff_dropout_cache': ff_dropout_cache,
            'ff_linear2_cache': ff_linear2_cache,
            'ff_h': ff_h,                     # 三维 relu 输出
            'ln2_cache': ln2_cache,
            'cos_emb': cos_emb, 'sin_emb': sin_emb,
            'num_heads': num_heads, 'head_dim': head_dim,
            'mask': mask,
            'dropout_rate': dropout_rate,
            'w_q': layer_params['w_q'], 'b_q': layer_params['b_q'],
            'w_k': layer_params['w_k'], 'b_k': layer_params['b_k'],
            'w_v': layer_params['w_v'], 'b_v': layer_params['b_v'],
            'w_o': layer_params['w_o'], 'b_o': layer_params['b_o'],
            'ff_w1': layer_params['ff_w1'], 'ff_b1': layer_params['ff_b1'],
            'ff_w2': layer_params['ff_w2'], 'ff_b2': layer_params['ff_b2'],
            'ln1_gamma': layer_params['ln1_gamma'], 'ln1_beta': layer_params['ln1_beta'],
            'ln2_gamma': layer_params['ln2_gamma'], 'ln2_beta': layer_params['ln2_beta'],
        }

    # ---- 输出层 ----
    x_flat = x.reshape(-1, embed_dim)
    logits_flat, output_linear_cache = linear(x_flat, params['output_w'], params['output_b'])
    logits = logits_flat.reshape(batch, seq_len, vocab_size)
    caches['output_linear_cache'] = output_linear_cache
    caches['x_final'] = x_flat   # 存储二维形式，便于 d_linear

    return logits, caches

# ========== 反向传播 ==========
def backward_transformer(dlogits, caches, params):
    grads = {}
    num_layers = params['num_layers']

    # ---- 输出层反向 ----
    dlogits_flat = dlogits.reshape(-1, params['vocab_size'])
    dx_flat, dw_out, db_out = d_linear(dlogits_flat, caches['output_linear_cache'])
    grads['output_w'] = dw_out
    grads['output_b'] = db_out

    # 将 dx_flat 重塑为三维，以便逐层传播
    batch = caches['one_hot'].shape[0]
    seq_len = caches['one_hot'].shape[1]
    embed_dim = params['embed_dim']
    dx = dx_flat.reshape(batch, seq_len, embed_dim)

    # ---- 从最后一层往前 ----
    for l in range(num_layers - 1, -1, -1):
        layer = caches[f'layer_{l}']
        x_in = layer['x_in']
        Q = layer['Q']; K = layer['K']; V = layer['V']
        Q_rot = layer['Q_rot']; K_rot = layer['K_rot']
        scores = layer['scores']
        attn = layer['attn']
        softmax_cache = layer['softmax_cache']
        dropout_cache = layer['dropout_cache']
        attn_linear_cache = layer['attn_linear_cache']
        ln1_cache = layer['ln1_cache']
        ff_linear1_cache = layer['ff_linear1_cache']
        ff_dropout_cache = layer['ff_dropout_cache']
        ff_linear2_cache = layer['ff_linear2_cache']
        ff_h = layer['ff_h']
        ln2_cache = layer['ln2_cache']
        cos_emb = layer['cos_emb']; sin_emb = layer['sin_emb']
        num_heads = layer['num_heads']; head_dim = layer['head_dim']

        # ---- LayerNorm2 反向 ----
        dx, dgamma2, dbeta2 = d_layer_norm(dx, ln2_cache)
        grads[f'layer_{l}_ln2_gamma'] = dgamma2
        grads[f'layer_{l}_ln2_beta'] = dbeta2

        # ---- FF 反向 ----
        # 将 dx 展平为二维，因为 d_linear 需要二维
        dx_flat = dx.reshape(-1, embed_dim)
        dh_flat, dw_ff2, db_ff2 = d_linear(dx_flat, ff_linear2_cache)
        grads[f'layer_{l}_ff_w2'] = dw_ff2
        grads[f'layer_{l}_ff_b2'] = db_ff2

        dh_flat = d_dropout(dh_flat, ff_dropout_cache)
        dh = dh_flat.reshape(batch, seq_len, embed_dim*4)
        dh = d_relu(dh, ff_h)  # dh 三维
        dh_flat = dh.reshape(-1, embed_dim*4)

        dx_ff_flat, dw_ff1, db_ff1 = d_linear(dh_flat, ff_linear1_cache)
        grads[f'layer_{l}_ff_w1'] = dw_ff1
        grads[f'layer_{l}_ff_b1'] = db_ff1

        dx_ff = dx_ff_flat.reshape(batch, seq_len, embed_dim)
        dx = dx_ff + dx

        # ---- LayerNorm1 反向 ----
        dx, dgamma1, dbeta1 = d_layer_norm(dx, ln1_cache)
        grads[f'layer_{l}_ln1_gamma'] = dgamma1
        grads[f'layer_{l}_ln1_beta'] = dbeta1

        # ---- 多头注意力反向 ----
        dx_flat = dx.reshape(-1, embed_dim)
        dx_attn_flat, dw_o, db_o = d_linear(dx_flat, attn_linear_cache)
        grads[f'layer_{l}_w_o'] = dw_o
        grads[f'layer_{l}_b_o'] = db_o

        dx_attn = dx_attn_flat.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        dV = np.matmul(attn.transpose(0, 1, 3, 2), dx_attn)
        dattn = np.matmul(dx_attn, V.transpose(0, 1, 3, 2))

        dattn = d_dropout(dattn, dropout_cache)
        dscores = d_softmax(dattn, softmax_cache)
        dscores = dscores / np.sqrt(head_dim)

        dK_rot = np.matmul(dscores.transpose(0, 1, 3, 2), Q_rot)
        dQ_rot = np.matmul(dscores, K_rot)

        dQ, dK = d_apply_rotary(dQ_rot, dK_rot, Q, K, cos_emb, sin_emb)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)

        # Q, K, V 的线性层反向（需要二维输入）
        dQ_flat = dQ.reshape(-1, embed_dim)
        dK_flat = dK.reshape(-1, embed_dim)
        dV_flat = dV.reshape(-1, embed_dim)

        dx_q_flat, dw_q, db_q = d_linear(dQ_flat, layer['q_cache'])
        dx_k_flat, dw_k, db_k = d_linear(dK_flat, layer['k_cache'])
        dx_v_flat, dw_v, db_v = d_linear(dV_flat, layer['v_cache'])

        grads[f'layer_{l}_w_q'] = dw_q
        grads[f'layer_{l}_b_q'] = db_q
        grads[f'layer_{l}_w_k'] = dw_k
        grads[f'layer_{l}_b_k'] = db_k
        grads[f'layer_{l}_w_v'] = dw_v
        grads[f'layer_{l}_b_v'] = db_v

        # 将三个梯度累加，并重塑为三维
        dx_q = dx_q_flat.reshape(batch, seq_len, embed_dim)
        dx_k = dx_k_flat.reshape(batch, seq_len, embed_dim)
        dx_v = dx_v_flat.reshape(batch, seq_len, embed_dim)
        dx = dx_q + dx_k + dx_v + dx

    # ---- 词嵌入反向 ----
    # dx 是最后一层输入（即第一层输入 x）的梯度，形状 (batch, seq, embed)
    # 词嵌入：x_flat = linear(one_hot_flat, embed_weight, bias=0)
    # 需要求 embed_weight 的梯度
    one_hot = caches['one_hot']
    dlogits_embed = dx.reshape(-1, embed_dim)  # (batch*seq, embed)
    # 利用 d_linear 的公式：dw = x.T @ dout
    one_hot_flat = one_hot.reshape(-1, params['vocab_size'])
    d_embed_weight = one_hot_flat.T @ dlogits_embed  # (vocab, embed)
    grads['embed_weight'] = d_embed_weight

    return grads

# ========== 参数初始化 ==========
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
