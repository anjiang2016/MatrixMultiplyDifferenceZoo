import numpy as np
data = np.load('tts_data.npz', allow_pickle=True)
samples = data['samples'].tolist()
print(samples[0]['mel'].shape)
print(samples[0]['text'])
