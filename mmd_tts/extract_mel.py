"""
梅尔频谱特征提取
从录制的语音数据中提取 log-mel 频谱，并保存为训练数据集。
用法：
    python extract_mel.py
依赖：
	pip install librosa numpy soundfile -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
"""
梅尔频谱特征提取 + 静音裁剪
用法：python extract_mel.py
"""

import os
import glob
import numpy as np
import librosa
import csv

# ===================== 配置 =====================
AUDIO_DIR = "./my_voice"
METADATA_FILE = "metadata.csv"
OUTPUT_FILE = "tts_data.npz"
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 256
FFT_SIZE = 1024
FMIN = 0
FMAX = 8000
ENERGY_THRESHOLD = -50  # dB 阈值，低于此值的帧视为静音

# ===================== 静音裁剪函数 =====================
def trim_silence_from_mel(mel, threshold=ENERGY_THRESHOLD):
    """
    mel: (T, n_mels)  log-mel 频谱 (dB 值)
    threshold: 静音阈值 (dB)，帧平均能量低于此值视为静音
    返回裁剪后的 mel (T', n_mels)
    """
    frame_energy = mel.mean(axis=1)  # (T,)
    valid_frames = np.where(frame_energy > threshold)[0]
    if len(valid_frames) == 0:
        # 如果全是静音，保留至少 1 帧避免空数据
        return mel[:1, :]
    start = valid_frames[0]
    end = valid_frames[-1] + 1
    return mel[start:end, :]

# ===================== 加载元数据 =====================
def load_metadata(csv_path):
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((row['file'], row['text']))
    print(f"📄 加载 {len(data)} 条记录")
    return data

# ===================== 提取梅尔频谱 =====================
def extract_log_mel(audio_path, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH,
                     fmin=FMIN, fmax=FMAX):
    y, _ = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        hop_length=hop_length,
        n_fft=FFT_SIZE,
        fmin=fmin, fmax=fmax
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = log_mel.T.astype(np.float32)  # (time, n_mels)
    return log_mel

# ===================== 构建分词器 =====================
def build_tokenizer_from_texts(texts):
    chars = sorted(set(''.join(texts)))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(chars)
    print(f"📝 词汇表大小: {vocab_size}")
    return char2idx, idx2char, vocab_size

# ===================== 主程序 =====================
def main():
    metadata = load_metadata(METADATA_FILE)
    if not metadata:
        print("⚠️ 未找到任何记录，请检查 metadata.csv 格式")
        return

    texts = [text for _, text in metadata]
    char2idx, idx2char, vocab_size = build_tokenizer_from_texts(texts)

    samples = []
    for fname, text in metadata:
        audio_path = os.path.join(AUDIO_DIR, fname)
        if not os.path.exists(audio_path):
            print(f"⚠️ 音频文件不存在: {audio_path}")
            continue

        # 提取原始 log-mel
        mel_raw = extract_log_mel(audio_path)
        print(f"原始 mel shape: {mel_raw.shape}")

        # ---- 裁剪静音 ----
        mel_trimmed = trim_silence_from_mel(mel_raw, threshold=ENERGY_THRESHOLD)
        print(f"裁剪后 mel shape: {mel_trimmed.shape}")

        # 文本编码
        text_ids = [char2idx[ch] for ch in text]

        samples.append({
            'text_ids': np.array(text_ids, dtype=np.int32),
            'mel': mel_trimmed,
            'text': text
        })
        print(f"✅ {fname}: mel {mel_trimmed.shape}, text length {len(text_ids)}")

    if not samples:
        print("❌ 没有成功提取任何样本")
        return

    np.savez(OUTPUT_FILE,
             samples=samples,
             char2idx=char2idx,
             idx2char=idx2char,
             vocab_size=vocab_size)
    print(f"\n🎉 已保存 {len(samples)} 个样本到 {OUTPUT_FILE}")

    # 统计信息
    mel_lens = [s['mel'].shape[0] for s in samples]
    print(f"📊 裁剪后梅尔帧数: 最小 {min(mel_lens)}, 最大 {max(mel_lens)}, 平均 {np.mean(mel_lens):.1f}")

if __name__ == "__main__":
    main()
