 使用说明
1. 安装依赖
bash
pip install sounddevice numpy scipy
2. 准备句子文件 sentences.txt
在脚本同目录下创建 sentences.txt，每行一句。例如：

text
大家好，我叫小明。
今天天气不错。
我想学习语音合成。
你喜欢听音乐吗？
3. 运行脚本
bash
python record_tts_data.py
4. 录音流程
脚本会逐句显示文本

按 Enter 开始录音（有 3 秒倒计时）

自动录制 4 秒（可修改 RECORD_SECONDS）

保存为 my_voice/voice_0001.wav 等

5. 生成元数据
录音完成后，会在当前目录生成 metadata.csv，格式：

text
file,text
voice_0001.wav,大家好，我叫小明。
voice_0002.wav,今天天气不错。
...
🔧 自定义参数
变量	说明	推荐值
SAMPLE_RATE	采样率	16000
RECORD_SECONDS	每句时长	2-5 秒
OUTPUT_DIR	音频保存目录	"./my_voice"
⚠️ 注意事项
录音环境尽量安静，麦克风距离适中

语速自然，不要刻意拖长或加快

如果句子较长，可适当增加 RECORD_SECONDS

录音后可用 Audacity 检查音量是否正常

🚀 下一步
录音完成后，运行 extract_mel.py 提取梅尔频谱，然后替换 create_sample 中的随机目标，开始训练你的 TTS 模型。
