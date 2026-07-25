import numpy as np
import sounddevice as sd
import queue
import threading
import time
import sys
import os

# 导入你的模型函数（假设 asr_train_10.py 中的函数已导出）
from asr_train import (
    asr_forward, init_asr_params, decode_ctc,
    normalize_mel, downsample_mel, load_asr_data
)

# ========== 音频参数 ==========
SAMPLE_RATE = 16000          # 采样率（Hz）
CHUNK_DURATION = 1.0         # 每次处理的音频长度（秒）
HOP_LENGTH = 160             # 帧移（10ms @ 16kHz）
N_FFT = 400                  # FFT 窗口大小（25ms）
N_MELS = 80                  # Mel 频带数
NUM_CHUNKS = 5               # 缓冲池大小（用于平滑识别，非必需）

# ========== 全局变量 ==========
audio_queue = queue.Queue()
stop_event = threading.Event()

def load_params(npz_path):
    """
    加载 .npz 格式的模型参数文件，返回参数字典。
    
    参数:
        npz_path: str, 模型文件路径（如 'asr_best.npz'）
    
    返回:
        params: dict, 包含所有模型参数的字典，键为参数名，值为 numpy 数组。
    
    异常:
        FileNotFoundError: 如果文件不存在。
        ValueError: 如果文件格式不正确或无法解析。
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        params = {key: data[key] for key in data.files}
        print(f"✅ 成功加载模型参数: {npz_path}")
        print(f"   包含 {len(params)} 个参数")
        return params
    except FileNotFoundError:
        print(f"❌ 错误: 文件 '{npz_path}' 不存在")
        raise
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        raise
# ========== 实时音频回调 ==========
def audio_callback(indata, frames, time_info, status):
    """音频回调函数，将数据放入队列"""
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

# ========== 音频流处理 ==========
def process_audio_stream(model_params, char_map):
    """
    实时处理音频流并输出识别结果
    """
    # 初始化 Mel 特征提取参数（使用 librosa 或自己实现）
    import librosa
    
    # 音频缓冲区（累积原始 PCM）
    audio_buffer = np.array([], dtype=np.float32)

    # 模型参数已经加载
    params = model_params

    print("🎤 实时听写已启动，请说话... (按 Ctrl+C 停止)")

    while not stop_event.is_set():
        try:
            # 从队列获取音频块（阻塞，超时0.1秒）
            data = audio_queue.get(timeout=0.1)
            audio_buffer = np.append(audio_buffer, data.flatten())
            
            # 当缓冲区长度达到 CHUNK_DURATION * SAMPLE_RATE 时进行处理
            if len(audio_buffer) >= int(CHUNK_DURATION * SAMPLE_RATE):
                # 截取固定长度
                chunk = audio_buffer[:int(CHUNK_DURATION * SAMPLE_RATE)]
                audio_buffer = audio_buffer[int(CHUNK_DURATION * SAMPLE_RATE):]
                # ---------- VAD 能量阈值检测 ----------
                rms = np.sqrt(np.mean(chunk**2))          # 均方根能量
                # 调试期间可打印：
                #print(f"RMS: {rms:.6f}", end='\r')
                ENERGY_THRESHOLD=0.01
                if rms < ENERGY_THRESHOLD:                # 阈值需要调整
                    print("\r识别结果: (静音)  ", end='', flush=True)
                    continue                              # 跳过本次识别，不送入模型 
                # 提取 Mel 频谱
                mel = librosa.feature.melspectrogram(
                    y=chunk, sr=SAMPLE_RATE, n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, n_mels=N_MELS
                )
                mel = librosa.power_to_db(mel, ref=np.max)  # 转换为 dB
                mel = mel.T  # 转置为 (T, n_mels)
                
                # ---------- 训练时的预处理顺序 ----------
                # 1. 下采样（与训练一致）
                #mel = downsample_mel(mel, factor=2)   # 假设训练时 factor=2

                # 2. 归一化（与训练一致）
                mel = normalize_mel(mel)

                # ---------- 推理时额外增加固定长度 ----------
                # 3. 固定为15帧（截断或填充）
                FIXED_LEN = 40
                if mel.shape[0] >= FIXED_LEN:
                    mel = mel[:FIXED_LEN, :]          # 取前15帧
                else:
                    pad = FIXED_LEN - mel.shape[0]
                    mel = np.pad(mel, ((0, pad), (0, 0)), mode='constant', constant_values=0) 
                # 添加 batch 维度
                mel = mel[None, :, :]  # (1, T', 80)
                
                # 模型推理
                input_length = np.array([mel.shape[1]], dtype=np.int32)
                # 目标留空（推理模式）
                logits, _, _, _ = asr_forward(params, mel, None, input_length, None)
                
                # CTC 解码
                decoded_ids = decode_ctc(logits, blank=0,merge_repeats=True,remove_blank=True)
                if decoded_ids:
                    text = ''.join([char_map.get(i, '?') for i in decoded_ids])
                    print(f"识别结果: {text}", end='\n', flush=True)
                else:
                    print("\r->", end='', flush=True)
                    
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break

    print("\n🛑 实时听写已停止")

# ========== 主程序 ==========
if __name__ == "__main__":
    # 加载训练好的模型参数
    # 假设你保存了最终的模型为 'asr_model_final.npz'
    params = load_params('asr_best_10.npz')  # 或者使用最终的模型文件
    
    # 加载词汇表（用于解码）
    _, _, idx2char, _ = load_asr_data('./tts_data.npz')  # 需要提供正确的路径
    
    # 启动音频流
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1)  # 100ms 块
    )
    
    with stream:
        try:
            process_audio_stream(params, idx2char)
        except KeyboardInterrupt:
            stop_event.set()
            print("\n程序退出")
