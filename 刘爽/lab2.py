import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
from scipy import stats
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义Type 2列的无关值（需要删除的内容）
INVALID_TYPE2_VALUES = {'A', 'BBB', '0', 'a', 'bbb', '0.0'}  # 包含大小写和可能的数值形式


# 检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


# 读取数据
def load_pokemon_data(file_path='Pokemon.csv'):
    try:
        # 先尝试自动检测编码
        encoding = detect_encoding(file_path)
        print(f"检测到的文件编码: {encoding}")

        # 尝试不同的编码
        encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for enc in encodings_to_try:
            try:
                print(f"尝试使用 {enc} 编码读取文件...")
                df = pd.read_csv(file_path, encoding=enc)
                print(f"成功使用 {enc} 编码读取文件，初始规模：{df.shape}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"使用 {enc} 编码时出错: {e}")
                continue

        # 如果所有编码都失败，使用错误处理方式
        print("所有编码尝试失败，使用错误处理方式读取...")
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
        return df

    except FileNotFoundError:
        print("文件未找到，请检查文件路径")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


# 数据概览
def data_overview(df):
    print("=" * 50)
    print("数据概览")
    print("=" * 50)
    print(f"数据集形状: {df.shape}")
    print("\n前5行数据:")
    print(df.head())

    print("\n数据基本信息:")
    print(df.info())

    print("\n描述性统计:")
    print(df.describe())

    print("\n缺失值统计:")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0])

    return df


# 数据类型检查
def check_data_types(df):
    print("\n" + "=" * 50)
    print("数据类型检查")
    print("=" * 50)

    # 检查列名，处理可能的编码问题
    print("列名:", df.columns.tolist())

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    print(f"数值型列: {list(numeric_columns)}")
    print(f"分类型列: {list(categorical_columns)}")

    return numeric_columns, categorical_columns


# 重复值检测
def detect_duplicates(df):
    print("\n" + "=" * 50)
    print("重复值检测")
    print("=" * 50)

    # 完全重复的行
    duplicate_rows = df[df.duplicated(keep=False)]
    if not duplicate_rows.empty:
        print(f"发现 {len(duplicate_rows)} 行完全重复的数据:")
        # 获取第一列和名称列的列名
        first_col = df.columns[0]
        name_col = 'Name' if 'Name' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        print(duplicate_rows[[first_col, name_col]].head(10))
    else:
        print("没有发现完全重复的行")

    # 名称重复的宝可梦
    name_col = 'Name' if 'Name' in df.columns else df.columns[1] if len(df.columns) > 1 else None
    if name_col:
        name_duplicates = df[df.duplicated(name_col, keep=False)]
        if not name_duplicates.empty:
            print(f"\n发现 {len(name_duplicates)} 行名称重复的宝可梦:")
            first_col = df.columns[0]
            print(name_duplicates[[first_col, name_col]].head(10))

    return duplicate_rows, name_duplicates


# 异常值检测 - 数值型数据
def detect_numeric_outliers(df, numeric_columns):
    print("\n" + "=" * 50)
    print("数值异常值检测")
    print("=" * 50)

    outlier_info = {}

    for col in numeric_columns:
        if col in df.columns:
            # 确保列是数值型
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # 使用IQR方法检测异常值
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            if not outliers.empty:
                print(f"\n{col} 列的异常值 (IQR方法):")
                print(f"正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"发现 {len(outliers)} 个异常值")
                outlier_info[col] = outliers

                # 显示部分异常值
                first_col = df.columns[0]
                name_col = 'Name' if 'Name' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                outlier_samples = outliers[[first_col, name_col, col]].head(5)
                for _, row in outlier_samples.iterrows():
                    print(f"  #{row[first_col]} {row[name_col]}: {row[col]}")

    return outlier_info


# 特殊值检测
def detect_special_values(df):
    print("\n" + "=" * 50)
    print("特殊值检测")
    print("=" * 50)

    special_cases = {}
    numeric_cols = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    # 检测0或负值
    for col in numeric_cols:
        if col in df.columns:
            # 确保列是数值型
            df[col] = pd.to_numeric(df[col], errors='coerce')
            zero_or_negative = df[df[col] <= 0]
            if not zero_or_negative.empty:
                print(f"\n{col} 列中的0或负值:")
                first_col = df.columns[0]
                name_col = 'Name' if 'Name' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                print(zero_or_negative[[first_col, name_col, col]].head())
                special_cases[f'{col}_zero_negative'] = zero_or_negative

    # 检测科学计数法表示的值
    for col in numeric_cols:
        if col in df.columns:
            # 检查是否包含科学计数法字符
            sci_notation = df[df[col].astype(str).str.contains('e|E', na=False)]
            if not sci_notation.empty:
                print(f"\n{col} 列中的科学计数法数值:")
                first_col = df.columns[0]
                name_col = 'Name' if 'Name' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
                print(sci_notation[[first_col, name_col, col]].head())
                special_cases[f'{col}_scientific'] = sci_notation

    return special_cases


# 逻辑一致性检查
def check_consistency(df):
    print("\n" + "=" * 50)
    print("逻辑一致性检查")
    print("=" * 50)

    issues = []

    # 确保数值列是数值类型
    stat_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    for col in stat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 检查Total列是否等于各属性之和
    if all(col in df.columns for col in ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']):
        calculated_total = df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].sum(axis=1)
        total_mismatch = df[abs(df['Total'] - calculated_total) > 1]
        if not total_mismatch.empty:
            print("Total列与各属性之和不匹配:")
            first_col = df.columns[0]
            name_col = 'Name' if 'Name' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
            print(total_mismatch[[first_col, name_col, 'Total']].head())
            issues.append('total_mismatch')

    # 检查Generation和Legendary列是否被置换
    if 'Generation' in df.columns and 'Legendary' in df.columns:
        # 检查Generation列中是否有布尔值
        gen_bool_like = df[df['Generation'].astype(str).str.upper().isin(['TRUE', 'FALSE'])]
        # 检查Legendary列中是否有数字
        leg_numeric = pd.to_numeric(df['Legendary'], errors='coerce').notna()
        legendary_numeric = df[leg_numeric]

        if not gen_bool_like.empty or not legendary_numeric.empty:
            print("\n可能存在Generation和Legendary列置换的情况:")
            first_col = df.columns[0]
            name_col = 'Name' if 'Name' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
            if not gen_bool_like.empty:
                print("Generation列中的布尔值:")
                print(gen_bool_like[[first_col, name_col, 'Generation', 'Legendary']].head())
            if not legendary_numeric.empty:
                print("Legendary列中的数值:")
                print(legendary_numeric[[first_col, name_col, 'Generation', 'Legendary']].head())
            issues.append('column_swap')

    return issues


# 可视化检测
def visualize_anomalies(df):
    print("\n" + "=" * 50)
    print("可视化异常检测")
    print("=" * 50)

    # 设置图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('宝可梦数据异常值可视化检测', fontsize=16, fontweight='bold')

    # 1. Attack属性的散点图
    if 'Attack' in df.columns:
        df['Attack'] = pd.to_numeric(df['Attack'], errors='coerce')
        axes[0, 0].scatter(range(len(df)), df['Attack'], alpha=0.6, color='red')
        axes[0, 0].set_title('Attack属性分布散点图')
        axes[0, 0].set_xlabel('数据索引')
        axes[0, 0].set_ylabel('Attack值')
        axes[0, 0].grid(True, alpha=0.3)

    # 2. 数值属性的箱线图
    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    available_numeric = []
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            available_numeric.append(col)

    if available_numeric:
        df[available_numeric].boxplot(ax=axes[0, 1])
        axes[0, 1].set_title('主要数值属性箱线图')
        axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Total属性的分布
    if 'Total' in df.columns:
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
        axes[0, 2].hist(df['Total'].dropna(), bins=30, alpha=0.7, color='green')
        axes[0, 2].set_title('Total属性分布直方图')
        axes[0, 2].set_xlabel('Total值')
        axes[0, 2].set_ylabel('频数')

    # 4. Type 2缺失情况
    type2_col = None
    for col in df.columns:
        if 'Type' in col and '2' in col:
            type2_col = col
            break

    if type2_col:
        type2_missing = df[type2_col].isna().value_counts()
        labels = ['有第二类型', '无第二类型'] if len(type2_missing) == 2 else ['有第二类型']
        axes[1, 0].pie(type2_missing.values, labels=labels, autopct='%1.1f%%')
        axes[1, 0].set_title('Type 2缺失情况')

    # 5. 重复值检测热图
    if len(available_numeric) > 1:
        correlation = df[available_numeric].corr()
        im = axes[1, 1].imshow(correlation, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('数值属性相关性热图')
        axes[1, 1].set_xticks(range(len(available_numeric)))
        axes[1, 1].set_yticks(range(len(available_numeric)))
        axes[1, 1].set_xticklabels(available_numeric, rotation=45)
        axes[1, 1].set_yticklabels(available_numeric)
        plt.colorbar(im, ax=axes[1, 1])

    # 6. 异常值汇总
    outlier_counts = {}
    for col in available_numeric:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_counts[col] = outliers

    axes[1, 2].bar(outlier_counts.keys(), outlier_counts.values(), color='orange')
    axes[1, 2].set_title('各属性异常值数量')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# 清洗前质量检查
def pre_clean_quality_check(df):
    print("\n" + "=" * 60)
    print("📊 数据清洗前质量检查")
    print("=" * 60)

    # 缺失值统计
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("1. 缺失值统计：")
        for col, cnt in missing.items():
            print(f"   - {col}: {cnt} 条（{cnt / len(df) * 100:.2f}%）")
    else:
        print("1. 缺失值统计：无缺失值")

    # Type 2列无关值提前检查
    if 'Type 2' in df.columns:
        # 预处理：去空格并转换为字符串
        df['Type 2'] = df['Type 2'].astype(str).str.strip()
        # 统计无关值数量
        invalid_type2 = df[df['Type 2'].isin(INVALID_TYPE2_VALUES)]
        if len(invalid_type2) > 0:
            print(f"2. Type 2列含无关值（{INVALID_TYPE2_VALUES}）的行：{len(invalid_type2)} 条（后续将删除）")

    # 重复行统计
    dup_cnt = df.duplicated().sum()
    print(f"3. 重复行数量：{dup_cnt} 条")

    return df


# 核心数据清洗函数（包含Type 2列无关值处理）
def clean_pokemon_data(df):
    print("\n" + "=" * 60)
    print("🧹 开始执行数据清洗")
    print("=" * 60)

    df_clean = df.copy()
    initial_rows = len(df_clean)
    print(f"初始数据行数：{initial_rows}")

    # 步骤1：处理Type 2列无关值
    print("\n1. Type 2列无关值处理：")
    if 'Type 2' in df_clean.columns:
        # 预处理：去空格并转换为字符串（避免空格导致误判）
        df_clean['Type 2'] = df_clean['Type 2'].astype(str).str.strip()
        # 标记包含无关值的行
        invalid_type2_mask = df_clean['Type 2'].isin(INVALID_TYPE2_VALUES)
        invalid_type2_cnt = invalid_type2_mask.sum()
        # 删除无关值所在行
        df_clean = df_clean[~invalid_type2_mask]
        print(f"   - 删除Type 2列含无关值（{INVALID_TYPE2_VALUES}）的行：{invalid_type2_cnt} 条，剩余行数：{len(df_clean)}")
    else:
        print("   - 数据中无Type 2列，跳过处理")

    # 步骤2：重命名首列为ID
    print("\n2. 列名标准化：")
    if df_clean.columns[0] != 'ID':
        original_first_col = df_clean.columns[0]
        df_clean.rename(columns={original_first_col: 'ID'}, inplace=True)
        print(f"   - 首列 '{original_first_col}' 重命名为 'ID'")
    else:
        print(f"   - 首列已为 'ID'，无需修改")

    # 步骤3：删除未定义行（排除Type 2列空值，但其无关值已在步骤1处理）
    print("\n3. 未定义行删除：")
    undefined_patterns = ['undefined', 'Undefined', 'UNDEFINED', np.nan, '']

    def is_invalid_row(row):
        for col_name, value in row.items():
            if col_name == 'Type 2':
                continue  # Type 2空值合法，无关值已删除
            if pd.isna(value) or str(value).strip() in undefined_patterns:
                return True
        return False

    invalid_mask = df_clean.apply(is_invalid_row, axis=1)
    invalid_cnt = invalid_mask.sum()
    df_clean = df_clean[~invalid_mask]
    print(f"   - 删除含未定义值的行：{invalid_cnt} 条，剩余行数：{len(df_clean)}")

    # 步骤4：删除完全空行
    print("\n4. 完全空行删除：")
    empty_row_cnt = df_clean.isnull().all(axis=1).sum()
    df_clean = df_clean.dropna(how='all')
    print(f"   - 删除完全空行：{empty_row_cnt} 条，剩余行数：{len(df_clean)}")

    # 步骤5：数值列类型修正
    print("\n5. 数值列类型修正：")
    numeric_columns = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'ID']
    existing_numeric_cols = [col for col in numeric_columns if col in df_clean.columns]

    for col in existing_numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    print(f"   - 已修正 {len(existing_numeric_cols)} 个数值列类型：{existing_numeric_cols}")

    # 步骤6：缺失值填充
    print("\n6. 缺失值填充：")
    missing_before = df_clean.isnull().sum().sum()

    # 数值列：中位数填充
    if existing_numeric_cols:
        imputer = SimpleImputer(strategy='median')
        df_clean[existing_numeric_cols] = imputer.fit_transform(df_clean[existing_numeric_cols])
        print(f"   - 数值列（{existing_numeric_cols}）：中位数填充")

    # 分类型列（除Type 2）：众数填充
    categorical_cols = ['Type 1', 'Legendary', 'Name']
    existing_cat_cols = [col for col in categorical_cols if col in df_clean.columns]
    for col in existing_cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"   - {col} 列：众数填充（填充值：{mode_val}）")

    # Type 2列：空值保留为空字符串（合法业务场景）
    if 'Type 2' in df_clean.columns:
        df_clean['Type 2'] = df_clean['Type 2'].fillna('')
        print(f"   - Type 2 列：空值保留为空字符串")

    missing_after = df_clean.isnull().sum().sum()
    print(f"   - 填充完成：填充前 {missing_before} 个缺失值 → 填充后 {missing_after} 个缺失值")

    # 步骤7：删除重复行
    print("\n7. 重复行处理：")
    duplicates_before = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    duplicates_after = df_clean.duplicated().sum()
    print(f"   - 删除重复行：{duplicates_before - duplicates_after} 条，剩余重复行：{duplicates_after} 条")

    # 步骤8：数据格式标准化
    print("\n8. 数据格式标准化：")
    # 文本列去空格
    text_cols = ['Name', 'Type 1', 'Type 2']
    existing_text_cols = [col for col in text_cols if col in df_clean.columns]
    for col in existing_text_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    print(f"   - 文本列去空格：{existing_text_cols}")

    # Legendary列统一为TRUE/FALSE
    if 'Legendary' in df_clean.columns:
        true_values = ['TRUE', 'True', 'true', '1', '1.0']
        df_clean['Legendary'] = df_clean['Legendary'].apply(
            lambda x: 'TRUE' if str(x).strip() in true_values else 'FALSE'
        )
        print(f"   - Legendary 列：统一为 TRUE/FALSE 格式")

    # 步骤9：删除完全空列
    print("\n9. 空列删除：")
    empty_cols_before = df_clean.isnull().all(axis=0).sum()
    df_clean = df_clean.loc[:, ~df_clean.isnull().all()]
    empty_cols_after = df_clean.isnull().all(axis=0).sum()
    print(f"   - 删除完全空列：{empty_cols_before - empty_cols_after} 列，剩余列数：{len(df_clean.columns)}")

    # 清洗结果汇总
    print("\n" + "=" * 60)
    print("✅ 数据清洗完成总结")
    print("=" * 60)
    final_rows = len(df_clean)
    print(f"初始行数：{initial_rows} → 最终行数：{final_rows}")
    print(f"总删除行数：{initial_rows - final_rows} 条（含Type 2无关值行）")
    print(f"最终数据规模：{df_clean.shape}")
    print(f"数据完整性：{((1 - df_clean.isnull().sum().sum() / df_clean.size) * 100):.2f}%")
    print(f"重复行数量：{df_clean.duplicated().sum()} 条")

    return df_clean


# 清洗后验证（重点验证Type 2列）
def post_clean_validation(df_cleaned, df_original):
    print("\n" + "=" * 60)
    print("🔍 清洗后数据验证")
    print("=" * 60)

    # Type 2列验证
    if 'Type 2' in df_cleaned.columns:
        remaining_type2 = df_cleaned['Type 2'].unique()
        invalid_remaining = [t for t in remaining_type2 if t in INVALID_TYPE2_VALUES]
        if len(invalid_remaining) == 0:
            print(f"1. Type 2列验证：已成功删除所有无关值（{INVALID_TYPE2_VALUES}）")
        else:
            print(f"1. Type 2列警告：仍存在无关值 {invalid_remaining}")

    # 其他基础验证
    print("2. 基础信息：")
    print(f"   - 数据规模：{df_cleaned.shape}")
    print(f"   - 缺失值：{df_cleaned.isnull().sum().sum()} 个")
    print(f"   - 重复行：{df_cleaned.duplicated().sum()} 个")


# 保存清洗后数据
def save_cleaned_data(df_cleaned, filename='pokemon_cleaned.csv'):
    df_cleaned.to_csv(filename, index=False)
    print(f"\n💾 清洗后数据已保存至：{filename}")


# 数据清理建议
def data_cleaning_recommendations(df, duplicate_rows, outlier_info, special_cases, consistency_issues):
    print("\n" + "=" * 50)
    print("数据清理建议")
    print("=" * 50)

    recommendations = []

    # 1. 重复数据处理建议
    if not duplicate_rows.empty:
        recommendations.append("建议删除完全重复的行")

    # 2. 异常值处理建议
    for col, outliers in outlier_info.items():
        if not outliers.empty:
            recommendations.append(f"{col}列中存在明显异常值，建议检查并修正")

    # 3. 特殊值处理建议
    for case_type, cases in special_cases.items():
        if not cases.empty:
            if 'zero_negative' in case_type:
                col = case_type.replace('_zero_negative', '')
                recommendations.append(f"{col}列中存在0或负值，需要检查")
            elif 'scientific' in case_type:
                col = case_type.replace('_scientific', '')
                recommendations.append(f"{col}列中存在科学计数法数值，建议转换为标准格式")

    # 4. 逻辑一致性建议
    if 'total_mismatch' in consistency_issues:
        recommendations.append("Total列与各属性之和不匹配，需要修正")
    if 'column_swap' in consistency_issues:
        recommendations.append("Generation和Legendary列可能存在置换，需要检查")

    # 5. Type 2异常处理
    type2_col = None
    for col in df.columns:
        if 'Type' in col and '2' in col:
            type2_col = col
            break

    if type2_col:
        # 检查Type 2中的异常值（如数字0）
        type2_anomalies = df[df[type2_col].astype(str).str.isdigit()]
        if not type2_anomalies.empty:
            recommendations.append("Type 2列中存在数值，建议清空这些值")

    # 输出建议
    if recommendations:
        print("发现以下数据质量问题，建议进行清理:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("数据质量良好，未发现明显问题")

    return recommendations


# 主函数
def main():
    # 加载数据
    df = load_pokemon_data()

    if df is None:
        print("无法读取数据文件，请检查文件路径和格式")
        return

    # 数据概览
    data_overview(df)

    # 数据类型检查
    numeric_cols, categorical_cols = check_data_types(df)

    # 重复值检测
    duplicate_rows, name_duplicates = detect_duplicates(df)

    # 异常值检测
    outlier_info = detect_numeric_outliers(df, numeric_cols)

    # 特殊值检测
    special_cases = detect_special_values(df)

    # 逻辑一致性检查
    consistency_issues = check_consistency(df)

    # 可视化检测
    visualize_anomalies(df)

    # 数据清理建议
    recommendations = data_cleaning_recommendations(
        df, duplicate_rows, outlier_info, special_cases, consistency_issues
    )

    # 清洗前质量检查
    df = pre_clean_quality_check(df)

    # 执行数据清洗
    df_cleaned = clean_pokemon_data(df)

    # 清洗后验证
    post_clean_validation(df_cleaned, df)

    # 保存清洗后数据
    save_cleaned_data(df_cleaned)

    print("\n" + "=" * 50)
    print("分析和清洗流程全部完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()