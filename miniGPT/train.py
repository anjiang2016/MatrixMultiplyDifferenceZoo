"""
train.py
--------
训练脚本（适配 YOLO 风格 model.py）
所有前向/反向逻辑在 model.py 中，训练循环在此调用。
"""

import sys
import os
import time
# 添加上级目录到 Python 路径，以便导入 funcs.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import glob
import numpy as np
from model import forward_transformer, backward_transformer, init_model_params
from funcs import softmax,d_softmax,cross_entropy, d_cross_entropy, adam_update
import time
import numpy as np
import os
def cosine_annealing_lr(epoch, total_epochs, lr_max=0.001, lr_min=1e-6):
    """余弦退火学习率"""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / total_epochs * np.pi))
def save_checkpoint(params, m, v, step, epoch, best_loss, losses, checkpoint_path='checkpoint.npz'):
    """保存完整训练状态（用于断点续训）"""
    np.savez(checkpoint_path,
             params=params,
             m=m,
             v=v,
             step=step,
             epoch=epoch,
             best_loss=best_loss,
             losses=losses)

def load_checkpoint(checkpoint_path='checkpoint.npz'):
    """加载检查点，返回 (params, m, v, step, start_epoch, best_loss, losses)"""
    if not os.path.exists(checkpoint_path):
        return None, None, None, 1, 0, float('inf'), []
    data = np.load(checkpoint_path, allow_pickle=True)
    params = data['params'].item()
    m = data['m'].item()
    v = data['v'].item()
    step = int(data['step'])
    start_epoch = int(data['epoch']) + 1  # 从下一epoch开始
    best_loss = float(data['best_loss'])
    losses = list(data['losses'])
    print(f"从检查点恢复：epoch={start_epoch}, step={step}, best_loss={best_loss:.4f}")
    return params, m, v, step, start_epoch, best_loss, losses
# ========== 分词器（纯函数） ==========
def build_tokenizer(chars, special_tokens=None):
    if special_tokens is None:
        special_tokens = ['<PAD>', '<UNK>', '<EOS>']
    all_tokens = special_tokens + chars
    char2idx = {c: i for i, c in enumerate(all_tokens)}
    idx2char = {i: c for i, c in enumerate(all_tokens)}
    return {
        'char2idx': char2idx,
        'idx2char': idx2char,
        'vocab_size': len(all_tokens),
        'pad_id': char2idx['<PAD>'],
        'unk_id': char2idx['<UNK>'],
        'eos_id': char2idx['<EOS>']
    }

def encode(tokenizer, text, max_len=None):
    char2idx = tokenizer['char2idx']
    unk_id = tokenizer['unk_id']
    pad_id = tokenizer['pad_id']
    tokens = []
    for c in text:
        tokens.append(char2idx.get(c, unk_id))
        if max_len and len(tokens) >= max_len:
            break
    if max_len is not None:
        if len(tokens) < max_len:
            tokens += [pad_id] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
    return tokens

def decode(tokenizer, indices):
    idx2char = tokenizer['idx2char']
    pad_id = tokenizer['pad_id']
    unk_id = tokenizer['unk_id']
    eos_id = tokenizer['eos_id']
    chars = []
    for i in indices:
        if i in (pad_id, unk_id, eos_id):
            continue
        chars.append(idx2char.get(i, ''))
    return ''.join(chars)

# ========== 数据提取 ==========
def extract_functions(source_dir):
    samples = []
    py_files = glob.glob(os.path.join(source_dir, '*.py'))
    for file in py_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = r'def\s+(\w+)\((.*?)\):(.*?)(?=\n\S|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        for name, params_str, body in matches:
            if name.startswith('_'):
                continue
            doc_match = re.search(r'"""(.*?)"""', body, re.DOTALL)
            doc = doc_match.group(1).strip() if doc_match else ''
            input_text = f"def {name}({params_str}):\n    \"\"\"{doc}\"\"\""
            output_text = f"def {name}({params_str}):\n{body.strip()}"
            samples.append((input_text, output_text))
    return samples

def build_vocab(samples):
    all_chars = set()
    for inp, out in samples:
        all_chars.update(inp + out)
    return sorted(all_chars)

def prepare_data(samples, tokenizer, max_len=512):
    data = []
    for inp, out in samples:
        full_text = inp + '\n' + out
        tokens = encode(tokenizer, full_text, max_len=max_len)
        if len(tokens) < 2:
            continue
        input_ids = np.array(tokens[:-1], dtype=np.int32)
        target_ids = np.array(tokens[1:], dtype=np.int32)
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = np.pad(input_ids, (0, pad_len), constant_values=tokenizer['pad_id'])
            target_ids = np.pad(target_ids, (0, pad_len), constant_values=tokenizer['pad_id'])
        else:
            input_ids = input_ids[:max_len-1]
            target_ids = target_ids[:max_len-1]
        data.append((input_ids, target_ids))
    return data
# ========== 训练步骤 ==========
def train_step(params, m,v,batch_inputs, batch_targets, step, lr):
    # 前向
    logits, caches = forward_transformer(params, batch_inputs)
    
    # ---- 计算损失（分开 softmax 和 cross_entropy） ----
    probs, softmax_cache = softmax(logits, axis=-1)   # 解包
    loss = cross_entropy(probs, batch_targets)         # 假设 cross_entropy 接受概率分布
    # ---- 反向 ----
    dprobs = d_cross_entropy(probs, batch_targets)     # 返回 dL/dprobs
    dlogits = d_softmax(dprobs, softmax_cache)         # 返回 dL/dlogits
    
    grads = backward_transformer(dlogits, caches, params)
    params, m,v,step = adam_update(params, m,v,grads, lr, step=step)
    return params, m,v,step, loss

def train_model(params, data, batch_size=16, epochs=10, lr_max=0.001, lr_min=1e-6,checkpoint_path='checkpoint.npz'):
    # 尝试加载检查点
    loaded = load_checkpoint(checkpoint_path)
    if loaded[0] is not None:
        params, m, v, step, start_epoch, best_loss, losses = loaded
    else:
        # 初始化Adam动量
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        step = 1
        start_epoch = 0
        best_loss = float('inf')
        losses = []

    indices = np.random.permutation(len(data))
    total_start_time = time.time()
    total_epochs = epochs
    for epoch in range(start_epoch, epochs):
        # 计算当前学习率
        lr_ca = cosine_annealing_lr(epoch, total_epochs, lr_max, lr_min)

        epoch_start = time.time()
        total_loss = 0
        num_batches = 0
        # 每个epoch重新打乱数据
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_inputs = np.stack([data[idx][0] for idx in batch_indices])
            batch_targets = np.stack([data[idx][1] for idx in batch_indices])
            params, m,v,step, loss = train_step(params, m,v,batch_inputs, batch_targets, step, lr_ca)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)
        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - total_start_time
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Epoch Time: {epoch_elapsed:.2f}s, Total Time: {total_elapsed:.2f}s")

        # ---- 保存最佳模型（仅参数） ----
        if avg_loss < best_loss:
            best_loss = avg_loss
            np.savez('best_model.npz', **params)  # 只存参数
            print(f"  -> Best model saved with loss {best_loss:.4f}")

            # ---- 保存检查点（全状态） ----
            save_checkpoint(params, m, v, step, epoch, best_loss, losses, checkpoint_path)
            # 可选：每 N 个 epoch 额外保存一次快照

    # 训练结束后，再保存一次最终模型（与检查点分离，只含参数）
    np.savez('last_model.npz', **params)
    print("训练完成。")
    return params
# ========== 主程序 ==========
def main():
    # 修改为你的 MMD 源码目录
    mmd_source_dir = '../'  # 示例，请根据实际情况修改
    if not os.path.exists(mmd_source_dir):
        print(f"错误：目录 {mmd_source_dir} 不存在")
        return

    samples = extract_functions(mmd_source_dir)
    print(f"提取到 {len(samples)} 个函数样本")
    if not samples:
        return

    chars = build_vocab(samples)
    tokenizer = build_tokenizer(chars)
    vocab_size = tokenizer['vocab_size']
    print(f"词汇表大小: {vocab_size}")
    # ---- 立即保存分词器（训练前） ----
    np.savez('tokenizer.npz',
             char2idx=tokenizer['char2idx'],
             idx2char=tokenizer['idx2char'],
             vocab_size=tokenizer['vocab_size'],
             pad_id=tokenizer['pad_id'],
             unk_id=tokenizer['unk_id'],
             eos_id=tokenizer['eos_id'])
    print("分词器已保存到 tokenizer.npz")
    data = prepare_data(samples, tokenizer, max_len=512)
    print(f"生成 {len(data)} 个训练样本")
    if not data:
        return

    params = init_model_params(
        vocab_size=vocab_size,
        embed_dim=64,
        num_layers=4,
        num_heads=8,
        max_seq_len=512,
        dropout_rate=0.1
    )
    print("模型参数初始化完成")
    np.savez('code_gen_params.npz', **params)
    print("模型参数已保存到 code_gen_params.npz")
    params = train_model(params, data, batch_size=8, epochs=60)

    np.savez('code_gen_params.npz', **params)
    print("模型参数已保存到 code_gen_params.npz")

if __name__ == '__main__':
    main()
