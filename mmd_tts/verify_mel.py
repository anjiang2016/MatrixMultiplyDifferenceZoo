"""
验证声码器：从数据集中读取真实梅尔频谱，合成音频
如果合成的音频清晰，说明声码器正常，问题在模型；如果合成的音频也是噪声，说明声码器或反归一化流程有问题。
用法：python verify_mel.py
"""

import numpy as np
import librosa
import soundfile as sf

def mel_to_audio(mel, sr=16000, n_mels=80, hop_length=256, n_iter=64,gain=10.0):
    """与 generate_audio.py 中相同的函数"""
    if mel.ndim == 3:
        mel = mel[0]
    # 反归一化（如果 mel 在 [-1, 1] 范围）
    if mel.min() >= -1.0 and mel.max() <= 1.0:
        mel = (mel + 1.0) / 2.0 * 80.0 - 80.0
    mel_spec = librosa.db_to_power(mel)
    mel_spec = mel_spec.T
    linear_spec = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr, power=2.0)
    linear_spec = linear_spec*10.0
    audio = librosa.griffinlim(linear_spec, hop_length=hop_length, n_iter=n_iter)
    return audio

def main():
    # 加载数据集
    data = np.load('tts_data.npz', allow_pickle=True)
    samples = data['samples'].tolist()

    # 取第一个样本
    sample = samples[0]
    mel = sample['mel']  # 原始 log-mel (T, 80)
    # ---- 归一化到 [-1, 1] ----
    mel_min = -80.0   # log-mel 的最小值（通常为 -80）
    mel_max = 0.0     # log-mel 的最大值（通常为 0）
    mel = (mel - mel_min) / (mel_max - mel_min) * 2.0 - 1.0
    # 现在 mel 的范围在 [-1, 1] 之间
    print(f"原始 mel 范围: [{mel.min():.4f}, {mel.max():.4f}]")
    print(f"原始 mel 均值: {mel.mean():.4f}")
    print(f"原始 mel 帧数: {mel.shape[0]}")
    print(f"🎵 梅尔频谱: {mel.shape}")
    # 合成音频
    audio = mel_to_audio(mel)
    sf.write('real_mel_output.wav', audio, 16000)
    print("✅ 已从真实梅尔频谱合成音频: real_mel_output.wav")

if __name__ == "__main__":
    main()
