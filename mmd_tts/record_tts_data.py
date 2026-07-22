"""
语音数据集录音脚本
用法：
    python record_tts_data.py
功能：
    1. 读取文本文件（每行一句），逐句提示用户朗读
    2. 录音（固定时长，默认 4 秒），保存为 16kHz 单声道 WAV
    3. 生成 metadata.csv 记录文件名和对应文本

依赖：
    pip install sounddevice numpy scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
"""

import os
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav

# ===================== 配置 =====================
SAMPLE_RATE = 16000        # 16kHz 采样率
RECORD_SECONDS = 4         # 每句录音时长（秒），可根据语速调整
OUTPUT_DIR = "./my_voice"  # 保存音频的文件夹
METADATA_FILE = "metadata.csv"
TEXT_FILE = "sentences.txt"  # 每行一句的文本文件

# 检查输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 读取句子列表 =====================
if not os.path.exists(TEXT_FILE):
    print(f"⚠️ 未找到 {TEXT_FILE}，请创建该文件，每行一句。")
    print("示例内容：")
    print("大家好，我叫小明。")
    print("今天天气不错。")
    print("我想学习语音合成。")
    exit(1)

with open(TEXT_FILE, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f if line.strip()]
print(f"📄 共加载 {len(sentences)} 个句子")

# ===================== 录音函数 =====================
def record_audio(duration, fs):
    """录制指定时长的音频"""
    print(f"🔴 录音中... ({duration}秒)")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # 等待录音完成
    print("✅ 录音结束")
    return audio.flatten()

# ===================== 主循环 =====================
metadata = []

for idx, text in enumerate(sentences):
    print("\n" + "=" * 50)
    print(f"[{idx+1}/{len(sentences)}] 请朗读：")
    print(f"   “{text}”")
    input("按 Enter 开始录音...")
    
    # 倒计时提示
    for i in range(3, 0, -1):
        print(f"{i}...", end=' ', flush=True)
        time.sleep(1)
    print("开始！")
    
    # 录音
    audio = record_audio(RECORD_SECONDS, SAMPLE_RATE)
    
    # 保存文件
    filename = f"voice_{idx+1:04d}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    write_wav(filepath, SAMPLE_RATE, audio)
    
    # 记录元数据
    metadata.append((filename, text))
    print(f"💾 已保存: {filepath}")

# ===================== 保存 metadata.csv =====================
with open(METADATA_FILE, 'w', encoding='utf-8') as f:
    f.write("file,text\n")
    for fname, txt in metadata:
        f.write(f"{fname},{txt}\n")

print("\n" + "=" * 50)
print(f"🎉 录音完成！共 {len(metadata)} 条")
print(f"📁 音频保存在 {OUTPUT_DIR}")
print(f"📄 元数据保存在 {METADATA_FILE}")
