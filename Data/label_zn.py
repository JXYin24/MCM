import pandas as pd
import numpy as np
import os
import csv  # 新增：用于模拟 CsvDataset 的读取逻辑
#  模拟 异常数据生成
# 配置
data_dir = '/data1/data_1/kyni/MCM/Data'
input_file = 'zntest_117.csv'  # 输入文件
output_file = 'zntest_117_ad.csv'  # 输出文件
anomaly_ratio = 0.1  # 异常比例,手动生成异常数据

# 构建输入路径
input_path = os.path.join(data_dir, input_file)

# 模拟 CsvDataset 的读取逻辑：读取 CSV，忽略 header，分离 X 和 y
x = []  # 特征列表
y = []  # 标签列表（原数据全为0）
with open(input_path, 'r') as data_from:
    csv_reader = csv.reader(data_from)
    next(csv_reader, None)  # 跳过 header（如果有）
    first_row = next(csv_reader, None)  # 读取第一数据行
    if first_row:
        data_dim = len(first_row)  # 动态计算 data_dim = 第一数据行的列数（假设无 label）
        print(f"调试: 从 CSV 第一数据行计算的 data_dim = {data_dim}")
        # 处理第一行
        if len(first_row) < data_dim:
            print(f"警告: 第一行不完整 (长度 {len(first_row)} < {data_dim})")
        else:
            try:
                features = [float(val) for val in first_row[0:data_dim]]
                x.append(features)
                y.append(0.0)  # 原数据 label=0
            except ValueError:
                print(f"警告: 第一行无效（非数值）: {first_row}")
    for row in csv_reader:  # 处理剩余行
        if len(row) < data_dim:  # 跳过不完整行
            print(f"警告: 跳过不完整行 (长度 {len(row)} < {data_dim}): {row}")
            continue
        try:
            features = [float(val) for val in row[0:data_dim]]  # 转换为 float
            x.append(features)
            y.append(0.0)  # 原数据 label=0
        except ValueError:
            print(f"警告: 跳过无效行（非数值）: {row}")
            continue

# 转换为 DataFrame 以便处理（X 是特征，y 是标签）
df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(data_dim)])
df['label'] = y

# 打印原数据信息
print(f"原数据形状: {df.shape}")
print("原数据前 5 行:")
print(df.head())
print("\n数值列: ", df.select_dtypes(include=[np.number]).columns.tolist())  # 显示哪些列是数值

# 生成异常（基于统计，只修改数值列）
num_anomalies = int(len(df) * anomaly_ratio)
anomaly_indices = np.random.choice(df.index, num_anomalies, replace=False)
anomaly_df = df.loc[anomaly_indices].copy()

# 只选择数值列计算 mean 和 std（排除 label）
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('label', errors='ignore')
means = df[numeric_cols].mean()
stds = df[numeric_cols].std()

# 修改数值列：添加偏移（2-4 倍 std 的随机噪声）
for col in numeric_cols:
    offset = np.random.uniform(2, 4) * stds[col]  # 随机偏移因子
    anomaly_df[col] += np.random.normal(loc=offset, scale=stds[col], size=len(anomaly_df))

anomaly_df['label'] = 1.0  # 异常 label=1（float）

# 合并并打乱
df_with_anomalies = pd.concat([df, anomaly_df], ignore_index=True)
df_with_anomalies = df_with_anomalies.sample(frac=1).reset_index(drop=True)

# 打印生成后信息（模拟 CsvDataset 的打印）
print(f"\n生成后数据形状: {df_with_anomalies.shape}")
print(f"异常值数量: {(df_with_anomalies['label'] == 1).sum()}")
print("生成后数据前 5 行（包含异常）:")
print(df_with_anomalies.head())
print(f"调试: 从生成数据计算的 dim = {len(df_with_anomalies.columns) - 1} (总列 {len(df_with_anomalies.columns)} - 1 for label)")

# 保存（不带 header，纯数据：特征 + label）
output_path = os.path.join(data_dir, output_file)
df_with_anomalies.to_csv(output_path, index=False, header=False)
print(f"\n已保存到: {output_path}（不包含表头，格式符合 CsvDataset）")