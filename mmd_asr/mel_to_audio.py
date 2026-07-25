import numpy as np
import librosa
import soundfile as sf
import os

# 1. 加载保存的 npz 文件
data = np.load('tts_data.npz', allow_pickle=True)
samples = data['samples'].tolist()
char2idx = data['char2idx'].item()
idx2char = data['idx2char'].item()

# 2. 选择您想听的样本（例如第一个）
index = 1  # 可修改
sample = samples[index]
mel = sample['mel']          # 形状 (T, n_mels)
text = sample.get('text', '未知')

print(f"样本 {index}: 文本 = '{text}', 帧数 = {mel.shape[0]}")

# 3. 将 log-mel (dB) 转换为功率谱（幅度）
# 因为 librosa 的 mel_to_audio 期望线性幅度谱或功率谱
mel_power = librosa.db_to_power(mel.T)  # 转置为 (n_mels, T)

# 4. 合成音频（Griffin-Lim）
sr = 16000
hop_length = 256
n_fft = 1024
audio_reconstructed = librosa.feature.inverse.mel_to_audio(
    mel_power,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_iter=64,          # 迭代次数，越大音质越好（但耗时）
    power=1.0           # 输入为幅度谱（power=1）或功率谱（power=2）
)

# 5. 保存为 WAV
output_file = f"reconstructed_{text}_{index}.wav"
sf.write(output_file, audio_reconstructed, sr)
print(f"✅ 合成音频已保存到: {output_file}")
