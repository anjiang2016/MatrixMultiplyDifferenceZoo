"""
语音生成脚本
用法：python generate_audio.py "大家好，我叫小明。"
"""

import sys
import numpy as np
import librosa
import soundfile as sf
from tts_train import init_tts_params,tts, load_sample_from_dataset,mse_loss
# ===== Griffin-Lim 合成函数 =====
def mel_to_audio(mel, sr=16000, n_mels=80, hop_length=256, n_iter=32):
    """与 generate_audio.py 中相同的函数"""
    if mel.ndim == 3:
        mel = mel[0]
    # 反归一化（如果 mel 在 [-1, 1] 范围）
    if mel.min() >= -1.0 and mel.max() <= 1.0:
        mel = (mel + 1.0) / 2.0 * 80.0 - 80.0
    mel_spec = librosa.db_to_power(mel)
    mel_spec = mel_spec.T
    linear_spec = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr, power=2.0)
    audio = librosa.griffinlim(linear_spec, hop_length=hop_length, n_iter=n_iter)
    return audio
# ===== 主函数 =====
def generate(text, model_path='tts_best.npz', output_path='output.wav'):
    # 加载分词器（从之前保存的数据中加载）

    sample = load_sample_from_dataset()
    char2idx = sample['char2idx']
    mel_target = sample['mel']
    vocab_size = int(sample['vocab_size'])

    # 编码文本
    text_ids = np.array([[char2idx.get(c, 0) for c in text]], dtype=np.int32)
    print(f"📝 文本: {text}")

    params = init_tts_params(vocab_size, embed_dim=256, num_heads=8, num_encoder_layers=3, mel_dim=80)
    # 加载模型参数
    data = np.load(model_path, allow_pickle=True)
    print(f"✅ 加载模型参数: {model_path}")
    best_params = {key: data[key].item() if data[key].dtype == np.dtype('O') else data[key] for key in data.files}
    best_loss = float(data['best_loss'])
    # 只更新模型权重，不改变结构参数
    for key in best_params:
        if key in params:
            params[key] = best_params[key]
    print("✅ 从 tts_best.npz 恢复模型参数")
    # 检查是否有 best_loss 字段
    if 'best_loss' in data:
        print(f"Saved best_loss: {data['best_loss']}")
    else:
        print("No best_loss field in file")

    # 推理
#pred_mel_,_ = tts(params, text_ids, mel_targets=mel_target, teacher_forcing=True)
    pred_mel_,_ = tts(params, text_ids, mel_targets=None, teacher_forcing=False,max_len=180)
#loss = mse_loss(pred_mel, target_mel)
    '''
    pred_mel = pred_mel_[0]
    print(f"🎵 生成梅尔频谱: {pred_mel.shape}")
    print("pred_mel 统计:")
    print(f"  mean: {pred_mel.mean():.6f}, std: {pred_mel.std():.6f}")
    print(f"  min: {pred_mel.min():.6f}, max: {pred_mel.max():.6f}")
    print(f"  帧间差异 (相邻帧差的绝对值平均): {np.abs(np.diff(pred_mel, axis=1)).mean():.6f}")
    mel_targets=mel_target[:, :-1, :]
    loss = mse_loss(pred_mel_, mel_targets)
    print(f"   loss:{loss:.6f}")
    print("\nmel_target 统计:")
    print(f"  mean: {mel_targets.mean():.6f}, std: {mel_targets.std():.6f}")
    print(f"  min: {mel_targets.min():.6f}, max: {mel_targets.max():.6f}")
    print(f"  帧间差异: {np.abs(np.diff(mel_targets, axis=1)).mean():.6f}")
    frame_corr = []
    for t in range(pred_mel_.shape[1]):
        corr = np.corrcoef(pred_mel_[0, t, :], mel_targets[0, t, :])[0, 1]
        frame_corr.append(corr)
    print(f"逐帧相关系数: min={min(frame_corr):.6f}, max={max(frame_corr):.6f}, mean={np.mean(frame_corr):.6f}")
    # 逐帧 MSE
    frame_mse = np.mean((pred_mel_ - mel_targets) ** 2, axis=(0, 2))
    print(f"Frame MSE: min={frame_mse.min():.6f}, max={frame_mse.max():.6f}, mean={frame_mse.mean():.6f}")
    '''
    # 合成音频
    audio = mel_to_audio(pred_mel_)
    sf.write(output_path, audio, 16000)
    print(f"✅ 音频保存至 {output_path}")

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "二"
    generate(text)
