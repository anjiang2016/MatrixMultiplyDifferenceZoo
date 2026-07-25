"""
ASR 训练脚本（单样本过拟合验证）
使用 Transformer 编码器 + CTC Loss
输入：梅尔频谱，输出：文本
依赖：上层目录的 funcs.py（需包含 ctc_loss, d_ctc_loss, softmax, linear, layer_norm, ...）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from funcs import (
    linear,  silu, d_silu, softmax, d_softmax,
    layer_norm, d_layer_norm, dropout, d_dropout,
    ctc_loss,d_ctc_loss,
    adam_update
)
def d_linear(dout, cache):
    """
    线性层反向传播
    dout: 上游梯度，形状与 linear 输出一致
    cache: (x, w, b) 来自 linear 前向
    返回: dx, dw, db
    """
    x, w, b = cache
    
    # 记录原始形状
    orig_shape = x.shape
    if x.ndim > 2:
        # 展平为 (batch*..., in_dim)
        x_flat = x.reshape(-1, x.shape[-1])
        dout_flat = dout.reshape(-1, dout.shape[-1])
    else:
        x_flat = x
        dout_flat = dout
    
    # 计算权重梯度
    dw = x_flat.T @ dout_flat  # (in_dim, out_dim)
    
    # 计算偏置梯度（如果存在）
    if b is not None:
        db = dout_flat.sum(axis=0)  # (out_dim,)
    else:
        db = None
    
    # 计算输入梯度
    dx_flat = dout_flat @ w.T  # (batch*..., in_dim)
    
    # 恢复形状
    if x.ndim > 2:
        dx = dx_flat.reshape(orig_shape)
    else:
        dx = dx_flat
    
    return dx, dw, db
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

# ========== 数据加载 ==========
def load_asr_data(npz_path='tts_data.npz'):
    data = np.load(npz_path, allow_pickle=True)
    samples = data['samples'].tolist()
    char2idx = data['char2idx'].item()
    idx2char = data['idx2char'].item()
    vocab_size = int(data['vocab_size'])
    print(f"加载 {len(samples)} 个样本，词汇表大小 {vocab_size}")
    return samples, char2idx, idx2char, vocab_size

def prepare_asr_batch(sample):
    mel = sample['mel']                  # (T, mel_dim)
    text_ids = sample['text_ids']        # (L,)
    ## 在目标序列前后插入空白（0）
    #text_ids = np.concatenate([text_ids_, [0]])
    mel = mel[None, :, :]                # (1, T, mel_dim)
    text_ids = text_ids[None, :]         # (1, L)

    #mel = downsample_mel(mel,2)
    input_length = np.array([mel.shape[1]], dtype=np.int32)
    target_length = np.array([text_ids.shape[1]], dtype=np.int32)
    return mel, text_ids, input_length, target_length

# ========== Transformer 编码器层 ==========
def transformer_encoder_layer(params, x, mask=None, layer_idx=0):
    batch, seq_len, embed_dim = x.shape
    num_heads = params['num_heads']
    head_dim = embed_dim // num_heads

    Q, q_cache = linear(x, params[f'enc_{layer_idx}_w_q'], params[f'enc_{layer_idx}_b_q'])
    K, k_cache = linear(x, params[f'enc_{layer_idx}_w_k'], params[f'enc_{layer_idx}_b_k'])
    V, v_cache = linear(x, params[f'enc_{layer_idx}_w_v'], params[f'enc_{layer_idx}_b_v'])

    Q = Q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    cos_emb = params['cos_emb'][:seq_len, :]
    sin_emb = params['sin_emb'][:seq_len, :]
    Q_rot, K_rot = apply_rotary(Q, K, cos_emb, sin_emb)

    scores = np.matmul(Q_rot, K_rot.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask * -1e9
    attn, softmax_cache = softmax(scores, axis=-1)
    attn_out = np.matmul(attn, V)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
    attn_out, attn_lin_cache = linear(attn_out, params[f'enc_{layer_idx}_w_o'], params[f'enc_{layer_idx}_b_o'])

    x = x + attn_out
    x, ln1_cache = layer_norm(x, params[f'enc_{layer_idx}_ln1_gamma'], params[f'enc_{layer_idx}_ln1_beta'])

    ff_h, ff1_cache = linear(x, params[f'enc_{layer_idx}_ff_w1'], params[f'enc_{layer_idx}_ff_b1'])
    ff_h_input = ff_h
    ff_h = silu(ff_h)   # 或用 silu
    ff_out, ff2_cache = linear(ff_h, params[f'enc_{layer_idx}_ff_w2'], params[f'enc_{layer_idx}_ff_b2'])
    x = x + ff_out
    x, ln2_cache = layer_norm(x, params[f'enc_{layer_idx}_ln2_gamma'], params[f'enc_{layer_idx}_ln2_beta'])

    cache = {
        'q_cache': q_cache, 'k_cache': k_cache, 'v_cache': v_cache,
        'Q': Q, 'K': K, 'V': V,
        'Q_rot': Q_rot, 'K_rot': K_rot,
        'softmax_cache': softmax_cache,
        'attn_lin_cache': attn_lin_cache,
        'ln1_cache': ln1_cache,
        'ff1_cache': ff1_cache,
        'ff_h_input': ff_h_input,
        'ff2_cache': ff2_cache,
        'ln2_cache': ln2_cache,
    }
    return x, cache

# ========== ASR 编码器 ==========
def asr_encoder(params, mel):
    batch, T, mel_dim = mel.shape
    embed_dim = params['embed_dim']
    x, proj_cache = linear(mel, params['input_proj_w'], params['input_proj_b'])
    caches = {'proj_cache': proj_cache}
    for l in range(params['num_layers']):
        x, layer_cache = transformer_encoder_layer(params, x, mask=None, layer_idx=l)
        caches[f'layer_{l}'] = layer_cache
    return x, caches

# ========== ASR 前向 ==========
def asr_forward(params, mel, targets, input_lengths, target_lengths):
    encoder_out, enc_caches = asr_encoder(params, mel)
    logits, lin_cache = linear(encoder_out, params['asr_output_w'], params['asr_output_b'])

#    loss, ctc_cache = ctc_loss(logits, targets, input_lengths, target_lengths, blank=0)
    # 将 lin_cache 放入 enc_caches 以便反向使用
    enc_caches['lin_cache'] = lin_cache

    loss = None

    ctc_cache = None
    if targets is not None:
#        # 转置为 (T, batch, vocab) 供 CTC
#        logits_ctc = logits.transpose(1, 0, 2)
        loss, ctc_cache = ctc_loss(logits, targets, input_lengths, target_lengths, blank=0)

    return logits, loss, ctc_cache, enc_caches

# ========== ASR 反向 ==========
def d_asr_encoder(dx, caches, params):
    grads = {}
    num_layers = params['num_layers']
    embed_dim = params['embed_dim']
    num_heads = params['num_heads']
    head_dim = embed_dim // num_heads

    for l in range(num_layers - 1, -1, -1):
        layer = caches[f'layer_{l}']
        dx_residual = dx
        dx, dgamma2, dbeta2 = d_layer_norm(dx, layer['ln2_cache'])
        grads[f'enc_{l}_ln2_gamma'] = dgamma2
        grads[f'enc_{l}_ln2_beta'] = dbeta2

        dx_ff, dw_ff2, db_ff2 = d_linear(dx, layer['ff2_cache'])
        grads[f'enc_{l}_ff_w2'] = dw_ff2
        grads[f'enc_{l}_ff_b2'] = db_ff2
        dx_ff = d_silu(dx_ff, layer['ff_h_input'])
        dx_ff, dw_ff1, db_ff1 = d_linear(dx_ff, layer['ff1_cache'])
        grads[f'enc_{l}_ff_w1'] = dw_ff1
        grads[f'enc_{l}_ff_b1']  = db_ff1
        dx = dx_ff + dx_residual

        dx, dgamma1, dbeta1 = d_layer_norm(dx, layer['ln1_cache'])
        grads[f'enc_{l}_ln1_gamma'] = dgamma1
        grads[f'enc_{l}_ln1_beta'] = dbeta1

        dx_attn, dw_o, db_o = d_linear(dx, layer['attn_lin_cache'])
        grads[f'enc_{l}_w_o'] = dw_o
        grads[f'enc_{l}_b_o'] = db_o

        batch, seq_len, _ = layer['q_cache'][0].shape
        dx_attn = dx_attn.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        V = layer['V']
        attn = layer['softmax_cache']
        dV = np.matmul(attn.transpose(0, 1, 3, 2), dx_attn)
        dattn = np.matmul(dx_attn, V.transpose(0, 1, 3, 2))

        dscores = d_softmax(dattn, layer['softmax_cache'])
        dscores = dscores / np.sqrt(head_dim)

        Q_rot = layer['Q_rot']; K_rot = layer['K_rot']
        dK_rot = np.matmul(dscores.transpose(0, 1, 3, 2), Q_rot)
        dQ_rot = np.matmul(dscores, K_rot)

        Q = layer['Q']; K = layer['K']
        cos_emb = params['cos_emb'][:seq_len, :]
        sin_emb = params['sin_emb'][:seq_len, :]
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

        dx = dx_q + dx_k + dx_v + dx_residual

    # 输入投影反向
    dx, dw_proj, db_proj = d_linear(dx, caches['proj_cache'])
    grads['input_proj_w'] = dw_proj
    grads['input_proj_b'] = db_proj

    return grads

def d_asr(dloss, ctc_cache, logits, enc_caches, params):
    dlogits = d_ctc_loss(dloss, ctc_cache, logits)
    dx, dw_out, db_out = d_linear(dlogits, enc_caches['lin_cache'])  # 注意：线性层的cache需保存
    enc_grads = d_asr_encoder(dx, enc_caches, params)
    grads = {'asr_output_w': dw_out, 'asr_output_b': db_out, **enc_grads}
    return grads

# ========== 模型初始化 ==========
def init_asr_params(vocab_size, mel_dim=80, embed_dim=256, num_layers=4, num_heads=8, max_seq_len=200):
    head_dim = embed_dim // num_heads
    cos_emb, sin_emb = get_rotary_embedding(max_seq_len, head_dim)
    params = {
        'embed_dim': embed_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'vocab_size': vocab_size,
        'mel_dim': mel_dim,
        'cos_emb': cos_emb,
        'sin_emb': sin_emb,
        'input_proj_w': np.random.randn(mel_dim, embed_dim) * 0.02,
        'input_proj_b': np.zeros(embed_dim),
        'asr_output_w': np.random.randn(embed_dim, vocab_size) * 0.02,
        'asr_output_b': np.zeros(vocab_size),
    }
    for l in range(num_layers):
        params[f'enc_{l}_w_q'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'enc_{l}_b_q'] = np.zeros(embed_dim)
        params[f'enc_{l}_w_k'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'enc_{l}_b_k'] = np.zeros(embed_dim)
        params[f'enc_{l}_w_v'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'enc_{l}_b_v'] = np.zeros(embed_dim)
        params[f'enc_{l}_w_o'] = np.random.randn(embed_dim, embed_dim) * 0.02
        params[f'enc_{l}_b_o'] = np.zeros(embed_dim)
        params[f'enc_{l}_ln1_gamma'] = np.ones(embed_dim)
        params[f'enc_{l}_ln1_beta'] = np.zeros(embed_dim)
        params[f'enc_{l}_ln2_gamma'] = np.ones(embed_dim)
        params[f'enc_{l}_ln2_beta'] = np.zeros(embed_dim)
        params[f'enc_{l}_ff_w1'] = np.random.randn(embed_dim, embed_dim*4) * 0.02
        params[f'enc_{l}_ff_b1'] = np.zeros(embed_dim*4)
        params[f'enc_{l}_ff_w2'] = np.random.randn(embed_dim*4, embed_dim) * 0.02
        params[f'enc_{l}_ff_b2'] = np.zeros(embed_dim)
    return params
def decode_ctc(logits, blank=0, merge_repeats=True, remove_blank=False):
    if logits.ndim == 3:
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    decoded = []
    prev = None
    for p in preds:
        if merge_repeats and p == prev:
            continue
        if remove_blank and p == blank:
            continue
        decoded.append(p)
        prev = p
    return decoded
def test_asr(params, mel, text_ids, idx2char, blank=0):
    """
    测试当前模型在给定音频上的识别结果
    """
    # 前向传播
    logits, _, _, _ = asr_forward(params, mel, None, None, None)
    # 解码（取第一个 batch）
    decoded_ids = decode_ctc(logits, blank=blank,merge_repeats=False,remove_blank=False)

    # 打印预测的 token ids
    print(f"预测 token ids: {decoded_ids}")
    print(f"真实 token ids: {text_ids[0].tolist()}")

    # 转换为字符
    pred_text = ''.join([idx2char.get(i, '?') for i in decoded_ids])
    
    # 真实文本
    true_text = ''.join([idx2char.get(i, '?') for i in text_ids[0]])
    return pred_text, true_text
def cosine_decay(epoch, total_epochs, lr_init=1e-4, lr_min=1e-6):
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(epoch / total_epochs * np.pi))
# 修改训练循环中的学习率计算
def warmup_cosine_decay(epoch, warmup_epochs=200, total_epochs=2000, lr_init=1e-4, lr_min=1e-8):
    if epoch < warmup_epochs:
        return lr_init * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(progress * np.pi))

def normalize_mel(mel, min_val=-80, max_val=0):
    return (mel - min_val) / (max_val - min_val) * 2 - 1
def denormalize_mel(normalized_mel, min_val=-80.0, max_val=0.0, clip=True):
    """
    将归一化到 [-1, 1] 的 mel 频谱还原为原始 dB 值。
    参数:
        normalized_mel: 归一化后的 mel，范围 [-1, 1]
        min_val: 原始的最小值（通常 -80）
        max_val: 原始的最大值（通常 0）
        clip: 是否将结果截断到 [min_val, max_val] 范围内（默认 True）
    返回:
        mel_db: 原始 dB 值，范围 [min_val, max_val]（若 clip=True）
    """
    mel_db = (normalized_mel + 1.0) / 2.0 * (max_val - min_val) + min_val
    if clip:
        mel_db = np.clip(mel_db, min_val, max_val)
    return mel_db
def trim_silence_from_mel(mel, threshold=-50):
    """
    裁剪 mel 频谱的静音帧（二维或三维）
    mel: (T, n_mels) 或 (1, T, n_mels)
    threshold: 每帧平均能量阈值（dB），低于该值视为静音
    返回: 裁剪后的 mel（保持原维度）
    """
    if mel.ndim == 3:
        # 取 batch 的第一个样本
        mel_2d = mel[0]
    else:
        mel_2d = mel
    # 计算每帧平均能量
    energy = mel_2d.mean(axis=1)  # (T,)
    valid = np.where(energy > threshold)[0]
    if len(valid) == 0:
        # 如果全静音，保留至少一帧
        if mel.ndim == 3:
            return mel[:, :1, :]
        else:
            return mel[:1, :]
    start = valid[0]
    end = valid[-1] + 1
    if mel.ndim == 3:
        return mel[:, start:end, :]
    else:
        return mel[start:end, :]
# 在 d_asr 返回 grads 后，应用梯度裁剪
def clip_grad_norm(grads, max_norm=1.0):
    total_norm = 0.0
    for g in grads.values():
        total_norm += np.sum(g ** 2)
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for key in grads:
            grads[key] *= scale
    return grads
'''
def ctc_loss(logits, targets, input_lengths, target_lengths, blank=0, eps=1e-12):
    """
    logits: (T, N, C) or (N, T, C) – 自动转置为 (T, N, C)
    """
    if logits.ndim == 3 and logits.shape[0] != input_lengths[0]:
        logits = logits.transpose(1, 0, 2)
    T, N, C = logits.shape

    # 稳定 log_softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    log_probs = logits - max_logits
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
    log_probs = np.clip(log_probs, -100, 0)   # 防止 -inf

    total_loss = 0.0
    caches = []
    for n in range(N):
        Tn = input_lengths[n]
        U = target_lengths[n]
        target = targets[n][:U]
        extended = [blank] + target.tolist() + [blank]
        L = len(extended)
        alpha = np.full((Tn+1, L), -np.inf, dtype=np.float64)
        alpha[0, 0] = 0.0

        log_probs_n = log_probs[:Tn, n, :]  # (Tn, C)

        for t in range(1, Tn+1):
            log_prob_t = log_probs_n[t-1]
            # 1) 同一位置
            alpha_t = alpha[t-1] + log_prob_t[extended]
            # 2) 前一个位置
            if L > 1:
                alpha_t[1:] = np.logaddexp(alpha_t[1:], alpha[t-1, :-1] + log_prob_t[extended[1:]])
            # 3) 跳过 blank（仅当标签不同且非 blank）
            for s in range(2, L):
                if extended[s] != blank and extended[s] != extended[s-2]:
                    val = alpha[t-1, s-2] + log_prob_t[extended[s-2]]
                    alpha_t[s] = np.logaddexp(alpha_t[s], val)
            alpha[t] = alpha_t

        log_likelihood = np.logaddexp(alpha[Tn, L-1], alpha[Tn, L-2])
        loss = -log_likelihood
        total_loss += loss
        caches.append({
            'alpha': alpha,
            'extended': extended,
            'L': L,
            'Tn': Tn,
            'log_probs_n': log_probs_n,
            'target': target,
            'blank': blank,
            'log_likelihood': log_likelihood,
            'n': n
        })
    avg_loss = total_loss / N
    return avg_loss, caches

def d_ctc_loss(dloss, cache, logits):
    if logits.ndim == 3 and logits.shape[0] != cache[0]['Tn']:
        logits = logits.transpose(1, 0, 2)
    T, N, C = logits.shape

    # 重新计算概率（用于梯度）
    max_logits = np.max(logits, axis=-1, keepdims=True)
    log_probs = logits - max_logits
    probs = np.exp(log_probs) / np.sum(np.exp(log_probs), axis=-1, keepdims=True)

    grad = np.zeros_like(logits, dtype=np.float64)
    for n, cache_n in enumerate(cache):
        alpha = cache_n['alpha']
        extended = cache_n['extended']
        L = cache_n['L']
        Tn = cache_n['Tn']
        blank = cache_n['blank']
        log_likelihood = cache_n['log_likelihood']
        log_probs_n = cache_n['log_probs_n']  # (Tn, C)

        # 后向 beta
        beta = np.full((Tn+1, L), -np.inf, dtype=np.float64)
        beta[Tn, L-1] = 0.0
        beta[Tn, L-2] = 0.0
        for t in range(Tn-1, -1, -1):
            log_prob_t = log_probs_n[t]
            beta_t = beta[t+1] + log_prob_t[extended]
            if L > 1:
                beta_t[:-1] = np.logaddexp(beta_t[:-1], beta[t+1, 1:] + log_prob_t[extended[:-1]])
            for s in range(L-2):
                if extended[s] != blank and extended[s] != extended[s+2]:
                    val = beta[t+1, s+2] + log_prob_t[extended[s+2]]
                    beta_t[s] = np.logaddexp(beta_t[s], val)
            beta[t] = beta_t

        # 梯度计算
        grad_n = np.zeros((Tn, C), dtype=np.float64)
        for t in range(Tn):
            log_unnorm = np.full(C, -np.inf, dtype=np.float64)
            for s, label in enumerate(extended):
                val = alpha[t, s] + beta[t, s]
                if val > -np.inf:
                    if log_unnorm[label] > -np.inf:
                        log_unnorm[label] = np.logaddexp(log_unnorm[label], val)
                    else:
                        log_unnorm[label] = val
            for k in range(C):
                if log_unnorm[k] > -np.inf:
                    diff = log_unnorm[k] - log_likelihood
                    if diff > 0:
                        diff = 0  # 数值安全
                    grad_n[t, k] = probs[t, n, k] - np.exp(diff)
                else:
                    grad_n[t, k] = probs[t, n, k]
        grad[:Tn, n, :] += grad_n

    grad *= dloss
    if grad.shape != logits.shape:
        grad = grad.transpose(1, 0, 2)
    return grad.astype(np.float32)
'''
def downsample_mel(mel, factor=2):
    """
    对 Mel 频谱在时间维度上进行隔帧采样。
    Args:
        mel: np.ndarray, 形状 (batch, T, mel_dim) 或 (T, mel_dim)
        factor: int, 下采样倍数（每 factor 帧取 1 帧）
    Returns:
        downsampled: 形状 (batch, T//factor, mel_dim) 或 (T//factor, mel_dim)
    """
    if mel.ndim == 2:
        # 单样本 (T, mel_dim)
        return mel[::factor, :]
    elif mel.ndim == 3:
        # 批量 (batch, T, mel_dim)
        return mel[:, ::factor, :]
    else:
        raise ValueError(f"Unsupported mel shape: {mel.shape}")
# ========== 训练循环 ==========
def train_asr():
    samples, char2idx, idx2char, vocab_size = load_asr_data('./tts_data.npz')
    # ---- 检查词汇表 ----
    print("=== 词汇表检查 ===")
    print(f"词汇表大小: {vocab_size}")
    # 打印所有字符及其索引
    for idx, char in sorted(idx2char.items()):
        print(f"  {idx}: '{char}'")
    print("==================")
    
    # 确认 blank 索引（通常为 0）
    blank = 0
    if blank in idx2char:
        print(f"blank 索引: {blank}, 对应字符: '{idx2char[blank]}'")
    else:
        print(f"警告: 索引 {blank} 不在词汇表中！")

    # 使用样本1（"二"）进行过拟合
    sample = samples[1]   # 确保索引1是"二"
    mel, text_ids, input_length, target_length = prepare_asr_batch(sample)
    #import pdb;pdb.set_trace()
    print(f" input_length : '{input_length}'  target_length: '{target_length}'")
    #mel=trim_silence_from_mel(mel,threshold=-45)
    #input_length[0] = mel.shape[1]
    mel = normalize_mel(mel)
    params = init_asr_params(vocab_size, mel_dim=mel.shape[-1], embed_dim=512, num_layers=8, num_heads=8, max_seq_len=mel.shape[1])
    #params = init_asr_params(vocab_size, mel_dim=mel.shape[-1], embed_dim=512, num_layers=8, num_heads=8, max_seq_len=15)

    epochs =600 
    lr_init = 2e-6
    best_loss = float('inf')
    step = 1

    for epoch in range(epochs):
        lr = warmup_cosine_decay(epoch, warmup_epochs=200, total_epochs=epochs, lr_init=lr_init, lr_min=5e-11)
        #lr = cosine_decay(epoch, total_epochs=epochs, lr_init=lr_init, lr_min=1e-8)
        logits, loss, ctc_cache, enc_caches = asr_forward(params, mel, text_ids, input_length, target_length)
        if not np.isfinite(loss) or np.isnan(loss):
            print(f"⚠️ 无效损失 {loss}，停止训练")
            break
        if np.any(~np.isfinite(logits)):
            print("⚠️ logits 包含 NaN/Inf，检查前向传播")
        dloss = 1.0
        grads = d_asr(dloss, ctc_cache, logits, enc_caches, params)
		# 在训练循环中，调用 adam_update 前：
        grads = clip_grad_norm(grads, max_norm=1.0)
        #for key in params:
        #    if key in grads:
        #        params[key] -= lr * grads[key]

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
            if loss < best_loss:
                best_loss = loss
                np.savez('asr_best.npz', **params)
                print(f"  -> Best model saved (loss={best_loss:.6f})")
        params, step = adam_update(params, grads, lr, step,beta1=0.9,beta2=0.99)
        # ---- 每100个epoch推理测试 ----
        if (epoch + 1) % 50 == 0:
            pred_text, true_text = test_asr(params, mel, text_ids, idx2char)
            print(f"  -> 预测: '{pred_text}' | 真实: '{true_text}'")
    print("训练完成！")

if __name__ == '__main__':
    train_asr()
