import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# -------------------------------------------------------------
# 1. 配置与数据加载
# -------------------------------------------------------------

# 获取脚本所在目录，确保路径正确
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'harth', 'harth') 

# HARTH 数据集采样率为 50Hz
SAMPLING_RATE = 50 

# 滑动窗口参数 (常用的HAR参数)
WINDOW_SIZE_S = 2.56  # 窗口大小 (秒)
STEP_SIZE_S = 1.28    # 步长 (秒)

WINDOW_SAMPLES = int(WINDOW_SIZE_S * SAMPLING_RATE)
STEP_SAMPLES = int(STEP_SIZE_S * SAMPLING_RATE)

print("--- 正在加载 HARTH 数据集 ---")

def load_all_data(data_path):
    """加载文件夹中所有受试者的 CSV 数据并合并"""
    # 确保路径存在
    if not os.path.exists(data_path):
        print(f"错误：路径 {data_path} 不存在。")
        return None
    
    # 使用 os.path.join 构建路径，确保跨平台兼容
    pattern = os.path.join(data_path, "S*.csv")
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"错误：在路径 {data_path} 中未找到任何 CSV 文件。")
        print(f"请检查路径是否正确。当前查找模式：{pattern}")
        # 列出目录中的文件以便调试
        if os.path.exists(data_path):
            files_in_dir = os.listdir(data_path)
            print(f"目录中的文件：{files_in_dir[:10]}...")  # 只显示前10个
        return None

    # 根据 HARTH 数据集信息，文件包含这些列：
    # timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label
    # 注意：实际文件中的标签列名为 'label'，需要重命名为 'Activity'
    
    list_ = []
    for filename in all_files:
        try:
            # 读取 CSV 文件（使用实际列名）
            df = pd.read_csv(filename, header=0)
            
            # 将 'label' 列重命名为 'Activity'（如果存在）
            if 'label' in df.columns:
                df = df.rename(columns={'label': 'Activity'})
            elif 'Activity' not in df.columns:
                print(f"警告：文件 {filename} 中未找到 'label' 或 'Activity' 列")
                continue
            
            # 添加受试者 ID（使用 os.path 处理跨平台路径）
            df['subject'] = os.path.basename(filename).split('.')[0]
            list_.append(df)
        except Exception as e:
            print(f"警告：无法加载或解析文件 {filename}。错误：{e}")
            continue

    if not list_:
        return None
        
    full_df = pd.concat(list_, axis=0, ignore_index=True)
    
    # HARTH 数据集的标签映射（根据数据集文档）
    # 1=Laying, 2=Sitting, 3=Standing, 4=Walking, 5=Running, 6=Stairs
    label_mapping = {
        '1': 'Laying',
        '2': 'Sitting', 
        '3': 'Standing',
        '4': 'Walking',
        '5': 'Running',
        '6': 'Stairs'
    }
    
    # 将标签转换为字符串并映射
    full_df['Activity'] = full_df['Activity'].astype(str).str.strip()
    full_df['Activity'] = full_df['Activity'].map(label_mapping).fillna(full_df['Activity'])
    
    # 过滤掉缺失值（如果存在）
    full_df.dropna(subset=['Activity'], inplace=True) 

    return full_df

try:
    df_raw = load_all_data(DATA_PATH)
    if df_raw is None:
        print("数据加载失败，程序退出。")
        exit()

    print(f"已加载 {df_raw.shape[0]} 个原始数据点。")
    print(f"发现的活动标签：{df_raw['Activity'].unique()}")
except Exception as e:
    print(f"加载数据时发生错误：{e}")
    import traceback
    traceback.print_exc()
    exit()


# -------------------------------------------------------------
# 2. 特征工程 (滑动窗口)
# -------------------------------------------------------------

print("\n--- 2. 特征工程：滑动窗口提取特征 ---")

def extract_features(window_data, activity_type):
    """从一个时间窗口中提取特征"""
    features = {}
    
    sensors = {'back': ['back_x', 'back_y', 'back_z'], 'thigh': ['thigh_x', 'thigh_y', 'thigh_z']}
    
    for sensor_name, columns in sensors.items():
        for col in columns:
            # 提取标准差 (Std) - 核心区分特征
            features[f'{sensor_name}_std_{col[-1]}'] = window_data[col].std()
            # 提取均值 (Mean)
            features[f'{sensor_name}_mean_{col[-1]}'] = window_data[col].mean()
            
        # 提取 Signal Magnitude Area (SMA) - 衡量运动剧烈程度
        features[f'{sensor_name}_SMA'] = (np.abs(window_data[columns[0]]) + 
                                          np.abs(window_data[columns[1]]) + 
                                          np.abs(window_data[columns[2]])).mean()
    
    features['Activity_Type'] = activity_type
    
    return pd.Series(features)

def sliding_window_feature_extraction(df):
    """对每个受试者应用滑动窗口"""
    all_features = []
    
    # 定义动/静态活动分组
    # 动态活动：Walking, Running, Stairs
    # 静态活动：Standing, Sitting, Laying
    dynamic_activities = ['Walking', 'Running', 'Stairs'] 
    static_activities = ['Standing', 'Sitting', 'Laying']
    
    for subject_id, subject_df in df.groupby('subject'):
        for start in range(0, len(subject_df) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            window = subject_df.iloc[start : start + WINDOW_SAMPLES]
            
            # 确定窗口的活动标签 (通常取众数)
            # 先过滤掉 NaN 值
            window_activities = window['Activity'].dropna()
            if len(window_activities) == 0:
                continue
                
            window_activity = window_activities.mode()
            if len(window_activity) == 0: 
                continue
            
            activity = window_activity.iloc[0] if isinstance(window_activity, pd.Series) else window_activity[0]
            
            # 归类为动态或静态
            if activity in dynamic_activities:
                activity_type = 'Dynamic Activities'
            elif activity in static_activities:
                activity_type = 'Static Activities'
            else:
                activity_type = 'Other'
                
            if activity_type != 'Other':
                features = extract_features(window, activity_type)
                features['subject'] = subject_id
                all_features.append(features)
        
    return pd.DataFrame(all_features)

try:
    df_features = sliding_window_feature_extraction(df_raw)

    if df_features is None or df_features.empty:
        print("错误：特征提取失败，未生成任何特征样本。")
        exit()

    print(f"特征提取完成，共生成 {df_features.shape[0]} 个特征样本。")
    print(f"提取的特征名称：{list(df_features.columns)[:5]}...")
except Exception as e:
    print(f"特征提取时发生错误：{e}")
    import traceback
    traceback.print_exc()
    exit()


# -------------------------------------------------------------
# 3. 统计分析与可视化
# -------------------------------------------------------------

print("\n--- 3. 统计分析：均值、标准差与对比 ---")

# 选择核心特征用于对比
comparison_features = [
    'back_std_x', 'back_std_y', 'back_std_z', 
    'thigh_std_x', 'thigh_std_y', 'thigh_std_z', 
    'back_SMA', 'thigh_SMA'
]

# 过滤掉非动静态活动的样本
df_compare = df_features[df_features['Activity_Type'].isin(['Dynamic Activities', 'Static Activities'])]

if df_compare.empty:
    print("错误：没有找到动态或静态活动的样本。")
    print(f"当前 Activity_Type 的唯一值：{df_features['Activity_Type'].unique()}")
    exit()

# 检查是否有足够的活动类型
available_types = df_compare['Activity_Type'].unique()
if 'Dynamic Activities' not in available_types or 'Static Activities' not in available_types:
    print(f"警告：缺少必要的活动类型。")
    print(f"可用的活动类型：{available_types}")
    print("将只显示可用类型的统计信息。")
    
    # 只计算可用的活动类型
    summary = df_compare.groupby('Activity_Type')[comparison_features].agg(['mean', 'std'])
    print("\n### 加速度特征统计表 ###")
    print(summary)
else:
    # 计算动静态活动的均值和标准差
    summary = df_compare.groupby('Activity_Type')[comparison_features].agg(['mean', 'std'])
    
    # 提取均值列（使用 xs 方法处理多级索引）
    mean_dynamic = summary.xs('mean', level=1, axis=1).loc['Dynamic Activities']
    mean_static = summary.xs('mean', level=1, axis=1).loc['Static Activities']
    
    # 计算差异
    mean_diff = mean_dynamic - mean_static
    
    # 格式化输出表格
    summary_table = pd.DataFrame({
        '动态活动均值': mean_dynamic.round(4),
        '静态活动均值': mean_static.round(4),
        '均值差异(动态-静态)': mean_diff.round(4),
    })
    
    print("\n### 加速度特征均值对比表 ###")
    print(summary_table)

# 绘制箱线图进行可视化对比
try:
    plt.figure(figsize=(15, 6))

    for i, feature in enumerate(comparison_features):
        plt.subplot(2, 4, i + 1)
        # 使用箱线图直观对比动静态活动下的特征分布
        sns.boxplot(x='Activity_Type', y=feature, data=df_compare)
        plt.title(feature.upper().replace('_', ' '), fontsize=10)
        plt.xlabel('')
        plt.xticks(rotation=45, ha='right')

    plt.suptitle('Dynamic vs Static Activities: Feature Distribution Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
except Exception as e:
    print(f"绘制图表时发生错误：{e}")
    import traceback
    traceback.print_exc()

print("\n--- 分析结论 ---")
print("在箱线图中，动态活动的 'std' (标准差) 和 'SMA' (信号幅值平均值) 通常远高于静态活动。")
print("这意味着这些特征能够有效地区分运动状态和静止状态。")