"""
train.py
--------
训练脚本（纯函数式，无类）
所有状态通过字典传递，适配 model.py 的前向+反向。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import glob
import numpy as np
from model import transformer_decoder, d_transformer_decoder, init_model_params
from funcs import cross_entropy, d_cross_entropy, adam_update

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
def train_step(params, batch_inputs, batch_targets, step, lr):
    # 前向
    logits, cache = transformer_decoder(params, batch_inputs)
    loss = cross_entropy(logits, batch_targets)
    # 反向
    dlogits = d_cross_entropy(logits, batch_targets)
    _, grads = d_transformer_decoder(dlogits, cache, params)
    # 更新
    params, step = adam_update(params, grads, lr, step=step)
    return params, step, loss

def train_model(params, data, batch_size=16, epochs=10, lr=0.001):
    indices = np.random.permutation(len(data))
    step = 1
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_inputs = np.stack([data[idx][0] for idx in batch_indices])
            batch_targets = np.stack([data[idx][1] for idx in batch_indices])
            params, step, loss = train_step(params, batch_inputs, batch_targets, step, lr)
            total_loss += loss
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return params

# ========== 主程序 ==========
def main():
    mmd_source_dir = '/Users/zhaomingming/本机文档/mydeepseekcode/matrix-multiply-zoo/'
#    mmd_source_dir = './mmd_lib'  # 替换为实际路径
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

    data = prepare_data(samples, tokenizer, max_len=512)
    print(f"生成 {len(data)} 个训练样本")
    if not data:
        return

    params = init_model_params(
        vocab_size=vocab_size,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        max_seq_len=512,
        dropout_rate=0.1
    )
    print("模型参数初始化完成")

    params = train_model(params, data, batch_size=16, epochs=10, lr=0.001)

    np.savez('code_gen_params.npz', **params)
    print("模型参数已保存到 code_gen_params.npz")

if __name__ == '__main__':
    main()
