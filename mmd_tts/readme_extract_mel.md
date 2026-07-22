使用步骤
安装依赖（如果尚未安装）

bash
pip install librosa numpy soundfile
准备文件

metadata.csv 已经在录音脚本中生成，格式为 file,text

./my_voice/ 目录下应包含所有 .wav 音频文件

运行提取脚本

bash
python extract_mel.py
输出文件

tts_data.npz：包含所有样本和分词器信息，可直接用于训练

📁 输出数据格式
tts_data.npz 包含以下键：

samples：列表，每个元素为 {'text_ids': ndarray, 'mel': ndarray, 'text': str}

char2idx：字符到索引的映射字典

idx2char：索引到字符的映射字典

vocab_size：词汇表大小

🔧 参数调整建议
参数	说明	推荐值
N_MELS	梅尔频带数	80（标准）
HOP_LENGTH	帧移	256（约 16ms）
FFT_SIZE	FFT 窗口	1024（约 64ms）
SAMPLE_RATE	采样率	16000
如果你的音频采样率不同，librosa.load 会自动重采样到 sr=SAMPLE_RATE。

🧪 测试单个样本
提取完成后，可以用以下代码快速验证一个样本：

python
import numpy as np
data = np.load('tts_data.npz', allow_pickle=True)
samples = data['samples'].tolist()
print(samples[0]['mel'].shape)
print(samples[0]['text'])
🚀 下一步
训练脚本现在可以直接加载 tts_data.npz，而不是使用随机生成的数据。你需要修改 create_sample 函数，改为从数据集中取一个样本，或者直接使用所有样本进行训练（多说话人或单说话人）。
