import os
import scipy.io as io
import numpy as np
import pandas as pd

# 配置
data_dir = './'  # 数据目录
dataset_name = 'arrhythmia'  # 修改为您的实际数据集名称，例如 'wbc'、'arrhythmia' 等

# 构建文件路径
path = os.path.join(data_dir, dataset_name + '.mat')

# 加载 .mat 文件
data = io.loadmat(path)
samples = data['X']  # 特征矩阵
labels = data['y'].flatten()  # 标签向量 (0: 正常, 1: 异常)

# 将数据转换为 Pandas DataFrame
num_features = samples.shape[1]
columns = [f'feature_{i}' for i in range(num_features)] + ['label']  # 特征列名 + 标签列
df = pd.DataFrame(np.hstack((samples, labels.reshape(-1, 1))), columns=columns)

# 打印 DataFrame 信息
print(f"DataFrame 形状: {df.shape}")
print(f"异常值比例: { (df['label'] == 1).sum() / len(df) * 100 :.2f}%")
print("\nDataFrame 前 5 行:")
print(df.head())

# 显示正常样本 (label=0) 的前 5 行
print("\n前 5 个正常样本 (label=0):")
print(df[df['label'] == 0].head())

# 显示异常样本 (label=1) 的前 5 行（如果存在）
print("\n前 5 个异常样本 (label=1):")
print(df[df['label'] == 1].head())

# 解释：异常值是 'label' 为 1 的行，通常是特征值偏离正常分布的样本。您可以进一步分析，如 df.describe() 或可视化。