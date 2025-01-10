import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 生成模拟数据
epochs = np.arange(1, 1301)  # Epoch数从1到1300

# 训练准确率模拟，使用逻辑增长函数模拟准确率的增长，并在接近1时趋于平稳
train_accuracy = 1 / (1 + np.exp(-0.05 * (epochs - 100)))  # 逻辑增长函数
train_accuracy += np.random.normal(0, 0.00001, epochs.shape)  # 添加少量噪声

# 测试准确率模拟，使用一个平滑的正弦波模拟波动，但波动幅度较小
test_accuracy = 0.55 + 0.01 * np.sin(2 * np.pi * epochs / 300)  # 平滑波动
test_accuracy += np.random.normal(0, 0.00001, epochs.shape)  # 添加少量噪声

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制训练准确率
plt.plot(epochs, train_accuracy, label='CIFAR10 Train Acc', color='blue')

# 绘制测试准确率
plt.plot(epochs, test_accuracy, label='CIFAR10 Test Acc', color='orange')

# 添加标题和标签
plt.title('curve_cnn_b32 Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()