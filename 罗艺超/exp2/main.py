import pandas as pd
import numpy as np
import chardet
import matplotlib.pyplot as plt

# 全局设置中文字体（以黑体为例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

#Windows -1252

# 使用检测到的编码读取文件
df = pd.read_csv("D:/doubao/lab2/Pokemon.csv", encoding='Windows-1252')

# # 3. 初步探索数据（了解数据基本结构）
# print("="*50)
# print("1. 数据集基本信息")
# print("="*50)
# print(f"数据集形状（行×列）：{df.shape}")  # 查看数据行数和列数
# print("\n数据集前5行：")
# print(df.head())  # 查看前5行，了解字段格式
# print("\n数据集末5行：")
# print(df.tail())  # 查看末5行
# print("\n数据集数据类型：")
# print(df.dtypes)  # 查看各字段的数据类型（重点关注Generation和Legendary是否正确）
# print("\n数据集缺失值统计：")
# print(df.isnull().sum())  # 查看各字段缺失值数量（Type 2允许缺失，因部分宝可梦无副属性）

#删除最后两行
df = df.iloc[:-2]
# print("\n再次查看数据集末5行：")
# print(df.tail())  # 查看末5行
type2_counts = df['Type 2'].value_counts(dropna=False)
# print("\n")
# print(type2_counts)

# 绘制柱状图
plt.figure(figsize=(10, 6))  # 设置图表大小（宽10，高6）
type2_counts.plot(kind='bar', color='skyblue')  # kind='bar' 表示柱状图，color设置颜色

# 添加标题和坐标轴标签
plt.title('Type 2 类别数量分布', fontsize=15)  # 标题
plt.xlabel('Type 2 类别', fontsize=12)  # x轴标签（类别名称）
plt.ylabel('数量', fontsize=12)  # y轴标签（计数）

# 调整x轴标签角度，避免重叠
plt.xticks(rotation=45, ha='right')  # rotation=45 旋转45度，ha='right' 右对齐

# 调整布局，避免标签被截断
plt.tight_layout()

# 显示图表
#plt.show()

df['Type 2'] = df['Type 2'].replace(['273', '0', 'A', 'BBB'],pd.NA)
type2_counts = df['Type 2'].value_counts(dropna=False)
# print("Type 2列的取值及数量统计（已转化指定值为NaN）：")
# print(type2_counts)

# 绘制柱状图
plt.figure(figsize=(10, 6))  # 设置图表大小（宽10，高6）
type2_counts.plot(kind='bar', color='skyblue')  # kind='bar' 表示柱状图，color设置颜色

# 添加标题和坐标轴标签
plt.title('Type 2 类别数量分布', fontsize=15)  # 标题
plt.xlabel('Type 2 类别', fontsize=12)  # x轴标签（类别名称）
plt.ylabel('数量', fontsize=12)  # y轴标签（计数）

# 调整x轴标签角度，避免重叠
plt.xticks(rotation=45, ha='right')  # rotation=45 旋转45度，ha='right' 右对齐

# 调整布局，避免标签被截断
plt.tight_layout()

# 显示图表
#plt.show()

# 标记重复行
duplicated_rows = df.duplicated()

# 标记所有重复行（包括首次出现的行，只要有重复就标记为True）
all_duplicated_rows = df.duplicated(keep=False)
# print(df[all_duplicated_rows])

# 查看是否存在重复行
has_duplicates = duplicated_rows.any()

if has_duplicates:
    print("存在重复行")
    num_duplicates = df.duplicated().sum()
    print(f"重复行的数量是:{num_duplicates}")
else  :
    print("不存在重复行")

#得到无重复行
df_new = df.drop_duplicates()
df=df_new


# 1. 提取Attack列，并转换为数值类型（无法转换的转为NaN）
# errors='coerce' 表示将非数值转换为NaN
attack_numeric = pd.to_numeric(df['Attack'], errors='coerce')

# 2. 按数值大小降序排序（保留原始索引，NaN会排在最后）
sorted_attack = attack_numeric.sort_values(ascending=False)

# 3. 输出结果
print("所有Attack值（含重复）按数值大小降序：")
print(sorted_attack)

# 1. 转换Attack列为数值型（无法转换的转为NaN）
df['Attack_numeric'] = pd.to_numeric(df['Attack'], errors='coerce')

# # 2. 过滤掉y轴中的NaN值（避免空值干扰）
# valid_data = df.dropna(subset=['Attack_numeric'])  # 只保留Attack_numeric非空的行
#
# # 3. 准备x轴（索引转为数值型）和y轴（确保是数值型）
# x = valid_data.index.astype(float)  # 强制转为float，避免被识别为分类
# y = valid_data['Attack_numeric'].astype(float)  # 确保是float类型
#
# # 4. 绘制散点图
# plt.figure(figsize=(12, 6))
# plt.scatter(x, y, color='skyblue', alpha=0.7, edgecolors='black')
#
# # 5. 设置标签和标题
# plt.title('Attack值的散点图', fontsize=15)
# plt.xlabel('记录索引（行号）', fontsize=12)
# plt.ylabel('Attack值', fontsize=12)
# plt.grid(linestyle='--', alpha=0.5)
#
# plt.tight_layout()
# plt.show()

#保留正常值，删除值过高的值
df_filtered = df[df['Attack_numeric'] <= 500]
x = df_filtered.index.astype(float)  # 强制转为float，避免被识别为分类
y = df_filtered['Attack_numeric'].astype(float)  # 确保是float类型


plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='skyblue', alpha=0.7, edgecolors='black')


plt.title('Attack值的散点图', fontsize=15)
plt.xlabel('记录索引（行号）', fontsize=12)
plt.ylabel('Attack值', fontsize=12)
plt.grid(linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# 尝试将 generation 列转换为数值类型，不能转换的可能存在异常
df['generation_numeric'] = pd.to_numeric(df['Generation'], errors='coerce')

# 定义 generation 的合理取值范围（这里假设是 1 到 8，可根据实际情况调整）
valid_generation_range = range(1, 9)

# 找出 generation 列不能转换为数值类型，或者转换后不在合理范围内，
# 且 Legendary 列看起来像数值的行
anomaly_mask = (df['generation_numeric'].isnull() | ~df['generation_numeric'].isin(valid_generation_range)) & \
               pd.to_numeric(df['Legendary'], errors='coerce').notnull()
anomaly_data = df[anomaly_mask]

print("可能存在异常的数据：")
print(anomaly_data)

# 直接对指定行（索引为 11 和 32）的两列值进行交换
df.loc[[11, 32], ['Generation', 'Legendary']] = df.loc[[11, 32], ['Legendary', 'Generation']].values

print(df.loc[[11, 32]])