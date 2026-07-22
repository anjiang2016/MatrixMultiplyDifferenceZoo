import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

def diagnose_mel_to_audio():
    # 1. 加载原始音频和其对应的梅尔频谱
    y_orig, sr = librosa.load('my_voice/voice_0001.wav', sr=16000)
    print(f"原始音频 RMS: {np.sqrt(np.mean(y_orig**2)):.6f}")
    print(f"原始音频峰值: {np.max(np.abs(y_orig)):.6f}")
    
    # 2. 加载梅尔频谱 (请确保 mel 是归一化前的 dB 值)
    data = np.load('tts_data.npz', allow_pickle=True)
    samples = data['samples'].tolist()
    mel = samples[0]['mel']  # (T, 80)
    print(f"Mel 范围: [{mel.min():.2f}, {mel.max():.2f}], 均值: {mel.mean():.2f}")
    
    # 3. 模拟 mel_to_audio 的转换过程，并打印每一步的能量
    # 3a. 如果是归一化后的 [-1,1]，先反归一化
    if mel.min() >= -1 and mel.max() <= 1:
        mel_db = (mel + 1) / 2 * 80 - 80
    else:
        mel_db = mel
    
    # 3b. dB 转功率谱
    mel_power = librosa.db_to_power(mel_db)
    print(f"Mel 功率谱均值: {mel_power.mean():.6f}, 最大值: {mel_power.max():.6f}")
    
    # 3c. 梅尔转线性 STFT (转置为 (n_mels, T))
    mel_power_T = mel_power.T
    linear_spec = librosa.feature.inverse.mel_to_stft(mel_power_T, sr=sr, power=2.0)
    print(f"线性 STFT 幅度谱均值: {linear_spec.mean():.6f}, 最大值: {linear_spec.max():.6f}")
    
    # 3d. Griffin-Lim 重建波形
    audio = librosa.griffinlim(linear_spec, hop_length=256, n_iter=64)
    print(f"重建音频 RMS (Griffin-Lim): {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"重建音频峰值: {np.max(np.abs(audio)):.6f}")
    
    # 4. 直接对比：跳过梅尔，直接用原始 STFT 做 Griffin-Lim 重建
    stft = librosa.stft(y_orig, n_fft=1024, hop_length=256)
    magnitude = np.abs(stft)
    audio_direct = librosa.griffinlim(magnitude, hop_length=256, n_iter=64)
    print(f"直接 STFT 重建 RMS: {np.sqrt(np.mean(audio_direct**2)):.6f}")
    print(f"直接 STFT 重建峰值: {np.max(np.abs(audio_direct)):.6f}")
    
    # 5. 保存对比文件
    sf.write('direct_stft_reconstruction.wav', audio_direct, sr)
    print("✅ 已保存直接 STFT 重建音频: direct_stft_reconstruction.wav")

if __name__ == "__main__":
    diagnose_mel_to_audio()
