import pandas as pd
import numpy as np
import os
import glob

# ---------------------------------------------------
# 2.4.1 读取数据集
# ---------------------------------------------------
# 确保工作目录正确（脚本所在目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 自动查找CSV文件（支持不同大小写）
csv_files = glob.glob('*.csv') + glob.glob('*.CSV')
pokemon_file = None
for f in csv_files:
    if 'pokemon' in f.lower():
        pokemon_file = f
        break

if pokemon_file is None:
    print("错误：未找到Pokemon.csv文件")
    print(f"当前目录: {os.getcwd()}")
    print(f"目录中的CSV文件: {csv_files}")
    raise FileNotFoundError("未找到Pokemon.csv文件，请确保文件在当前目录下")

# 尝试不同的编码方式
try:
    df_pokemon = pd.read_csv(pokemon_file, encoding='GBK')
except UnicodeDecodeError:
    try:
        df_pokemon = pd.read_csv(pokemon_file, encoding='utf-8')
    except:
        df_pokemon = pd.read_csv(pokemon_file, encoding='latin-1')
print("原始宝可梦数据集（前5行）:")
print(df_pokemon.head())
print("\n数据集信息:")
df_pokemon.info()
#

# ---------------------------------------------------
# 2.4.2 完整性：缺失值处理
# ---------------------------------------------------
# 1. 检查缺失值
print("\n每列的缺失值数量:")
print(df_pokemon.isnull().sum())
#

# 2. 缺失值填充：'Type 2' 是最常出现缺失值的列（代表双属性宝可梦的第二属性）。
#    使用 'None' 字符串进行填充，表示该宝可梦为单属性。
df_pokemon['Type 2'] = df_pokemon['Type 2'].fillna('None')
print("\n填充 Type 2 缺失值后的缺失值数量:")
print(df_pokemon.isnull().sum())
#

# 3. 如果 'Name' 或 'Type 1' 存在少量缺失值，可以选择删除对应行
# df_pokemon.dropna(subset=['Name', 'Type 1'], inplace=True)
#

# ---------------------------------------------------
# 2.4.3 一致性：重复值处理
# ---------------------------------------------------
# 1. 检查重复行：基于所有列检查完全重复的记录
print("\n完全重复行数量:", df_pokemon.duplicated().sum())
#

# 2. 删除重复行：基于所有列删除完全重复的记录
df_pokemon.drop_duplicates(inplace=True)
print("删除重复行后的数据集大小:", len(df_pokemon))
#

# 3. (可选) 检查 'Name' 列的名称重复 (排除 Mega 进化等特殊情况)
print("\n基于 Name 的重复数量 (用于检查数据一致性):", df_pokemon['Name'].duplicated().sum())

# ---------------------------------------------------
# 2.4.4 准确性：异常值处理 (以 'Attack' 攻击力为例，使用 IQR 方法)
# ---------------------------------------------------
STAT_COL = 'Attack' # 选择一个核心数值属性进行异常值处理

# 0. 先将数值列转换为数值类型（如果当前是object类型）
numeric_cols = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']
for col in numeric_cols:
    if col in df_pokemon.columns:
        df_pokemon[col] = pd.to_numeric(df_pokemon[col], errors='coerce')

# 1. 计算 Q1 (25th percentile) 和 Q3 (75th percentile)
Q1 = df_pokemon[STAT_COL].quantile(0.25)
Q3 = df_pokemon[STAT_COL].quantile(0.75)
IQR = Q3 - Q1
#

# 2. 定义异常值的边界
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
#

# 3. 标记并处理异常值 (将异常值替换为边界值)
df_pokemon[STAT_COL + '_cleaned'] = df_pokemon[STAT_COL].apply(
    lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x)
)
#

# 比较处理前后的统计信息
print(f"\n'{STAT_COL}' 原始统计信息:")
print(df_pokemon[STAT_COL].describe())
print(f"\n'{STAT_COL + '_cleaned'}' 处理后统计信息:")
print(df_pokemon[STAT_COL + '_cleaned'].describe())
#

# ---------------------------------------------------
# 2.4.5 时效性/有效性：数据格式和逻辑检查
# ---------------------------------------------------
# 宝可梦数据集缺少时间字段，此步骤改为检查数据有效性和格式
# 1. 检查 'Legendary' (传说宝可梦) 列，确保其为布尔值或二元整数
#    (如果原始数据是 True/False 字符串，需要转换)
df_pokemon['Legendary'] = df_pokemon['Legendary'].astype(bool)
print("\n'Legendary' 数据类型转换完成:", df_pokemon['Legendary'].dtype)
#

# 2. 检查 'Generation' 列的逻辑范围 (世代编号通常为 1 到 9)
generation_min = df_pokemon['Generation'].min()
generation_max = df_pokemon['Generation'].max()
print(f"Generation 范围: {generation_min} 到 {generation_max}")

# 3. 示例：修正超出逻辑范围的 'Generation' 值（假设有效范围是 1 到 9）
df_pokemon['Generation'] = df_pokemon['Generation'].clip(lower=1, upper=9) 
#

print("\n数据质量处理后的数据集结构:")
print(df_pokemon.info())

# ---------------------------------------------------
# 保存清洗后的数据
# ---------------------------------------------------
output_file = 'clean_data.csv'
df_pokemon.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n清洗后的数据已保存为: {output_file}")
print(f"保存的数据行数: {len(df_pokemon)}")