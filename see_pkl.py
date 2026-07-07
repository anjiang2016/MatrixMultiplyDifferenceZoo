import pickle
import matplotlib.pyplot as plt

with open('/Users/zhaomingming/data_sets/mnist/mnist.pkl', 'rb') as f:
    data = pickle.load(f)

# 先打印数据结构，帮助判断
print(f"数据类型: {type(data)}")
print(f"数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")

# 如果是列表，查看前几个元素的结构
if isinstance(data, list):
    print(f"第一个元素类型: {type(data[0])}")
    print(f"第一个元素: {data[0] if not isinstance(data[0], (np.ndarray, list)) else 'array/list'}")
    
# 尝试显示第一张图（需要根据实际结构调整）
# 这里需要你根据打印结果手动调整
