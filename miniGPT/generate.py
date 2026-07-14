"""
generate.py - 代码生成（使用训练好的模型）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model import forward_transformer
from funcs import softmax

# ========== 加载分词器 ==========
def load_tokenizer(tokenizer_path='tokenizer.npz'):
    """加载训练时保存的分词器"""
    data = np.load(tokenizer_path, allow_pickle=True)
    tokenizer = {
        'char2idx': data['char2idx'].item(),
        'idx2char': data['idx2char'].item(),
        'vocab_size': int(data['vocab_size']),
        'pad_id': int(data['pad_id']),
        'unk_id': int(data['unk_id']),
        'eos_id': int(data['eos_id'])
    }
    print("✅ 分词器加载成功")
    return tokenizer

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

# ========== 模型加载 ==========
def load_model_params(model_path='best_model.npz'):
    """加载模型参数（过滤掉动量项）"""
    data = np.load(model_path, allow_pickle=True)
    params = {key: data[key].item() if data[key].dtype == np.dtype('O') else data[key] for key in data.files}
    # 只保留非动量键
#    filtered = {k: v for k, v in params.items() if not k.endswith('_m') and not k.endswith('_v')}
#    print(f"✅ 模型加载成功，共 {len(filtered)} 个参数组")
#    return filtered
    return params
# ========== 生成函数 ==========
def generate_code(params, prompt, tokenizer, max_new_tokens=200, temperature=0.8):
    """自回归生成代码"""
    input_ids = np.array([encode(tokenizer, prompt)], dtype=np.int32)
    for _ in range(max_new_tokens):
        logits, _ = forward_transformer(params, input_ids)
        next_logits = logits[0, -1, :] / temperature
        probs, _ = softmax(next_logits, axis=0)   # softmax返回(out, cache)
        next_token = np.random.choice(len(probs), p=probs)
        input_ids = np.concatenate([input_ids, np.array([[next_token]])], axis=1)
        if next_token == tokenizer['eos_id']:
            break
    generated = decode(tokenizer, input_ids[0])
    return generated

# ========== 主程序 ==========
def main():
    # 1. 加载分词器
    if not os.path.exists('tokenizer.npz'):
        print("❌ 未找到 tokenizer.npz，请先运行训练脚本以保存分词器。")
        return
    tokenizer = load_tokenizer()

    # 2. 加载模型
    params = load_model_params('best_model.npz')
    # 3. 生成代码
    prompt = "def conv("  # 可以修改为其他提示
    generated = generate_code(params, prompt, tokenizer, max_new_tokens=200, temperature=0.8)
    print("\n生成的代码：")
    print(generated)

    # 可选：保存生成结果到文件
    with open('generated_code.txt', 'w') as f:
        f.write(generated)

if __name__ == '__main__':
    main()
