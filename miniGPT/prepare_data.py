"""
train.py
--------
训练脚本（适配 YOLO 风格 model.py）
所有前向/反向逻辑在 model.py 中，训练循环在此调用。
"""

import sys
import time
import re
import glob
import numpy as np
import os
from train import build_tokenizer,encode,extract_functions,build_vocab,prepare_data
from generate import decode

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
    data = prepare_data(samples, tokenizer, max_len=512)
    print(f"生成 {len(data)} 个训练样本")
    lengths =[len(inp+ '\n' + out) for inp,out in samples]
    print(f"Max length: {max(lengths)}, Min: {min(lengths)}, Avg: {np.mean(lengths)}")
    for i, (inp,tgt) in enumerate(data):
        print(f"{i}: 开头 {decode(tokenizer, inp[:20])} | 结尾 {decode(tokenizer, tgt[-20:])}")
    '''
# 在 train.py 的 main() 中，data = prepare_data(...) 之后添加：
    for idx in range(len(data)):
        sample_input, sample_target = data[idx]
        print("\n=== 训练数据检查 ===")
        print("输入序列前 20 个 token ID:", sample_input[:20])
        print("目标序列前 20 个 token ID:", sample_target[:20])
        print("\n解码后的输入开头:")
        print(repr(decode(tokenizer, sample_input[:100])))
        print("\n解码后的目标开头:")
        print(repr(decode(tokenizer, sample_target[:100])))
        print("\n输入序列末尾 10 个 token ID:", sample_input[-10:])
        print("目标序列末尾 10 个 token ID:", sample_target[-10:])
        print("解码后的输入末尾:")
        print(repr(decode(tokenizer, sample_input[-50:])))
        print("解码后的目标末尾:")
        print(repr(decode(tokenizer, sample_target[-50:])))
    '''
if __name__ == '__main__':
    main()
