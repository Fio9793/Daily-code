import pandas as pd
import numpy as np
# 直接使用GBK编码（中文Windows系统常用）
data = pd.read_csv("data.csv", encoding='gbk')
#过滤空行
data_cleaned=data.dropna(how='any')
#过滤traffic！=0 f一般
filtered_data1 = data_cleaned.loc[(data_cleaned['traffic'] != 0) ]
filtered_data=filtered_data1.loc[(data_cleaned['from_level'] == '一般节点')]
# 显示原始数据
print(data)
#显示清理空行的数据
print(data_cleaned)
#
print(filtered_data)
#加权抽象
data_before_sample=filtered_data.copy()
columns=data_before_sample.columns
weight_sample=data_before_sample.copy()
#添加权重行
weight_sample['weight']=0
#设置权重
for i in weight_sample.index:
    if weight_sample.at[i,'to_level']=='一般节点':
        weight=1
    else:
        weight=5
    weight_sample.at[i,'weight']=weight
#抽取50个样本
weight_sample_finish=weight_sample.sample(n=50,weights='weight')
weight_sample_finish=weight_sample_finish[columns]
print(weight_sample_finish)
#随机抽样
random_sample=filtered_data
random_sample_finish=random_sample.sample(n=50)
random_sample_finish=random_sample_finish[columns]
print(random_sample_finish)
#分层抽样 一般节点17 网络核心节点33
ybjd=filtered_data.loc[filtered_data['to_level']=='一般节点']
wlhx=filtered_data.loc[filtered_data['to_level']=='网络核心']
fc_sample=pd.concat([ybjd.sample(17),wlhx.sample(33)])
print(fc_sample)

# 系统抽样（等距抽样）
def systematic_sampling(data, n):
    N = len(data)
    k = N // n  # 计算抽样间隔
    # 随机选择起始点
    start = np.random.randint(0, k)
    # 选择样本索引
    indices = [start + i * k for i in range(n) if (start + i * k) < N]
    return data.iloc[indices]

# 进行系统抽样
systematic_sample = systematic_sampling(data_before_sample, 50)
systematic_sample = systematic_sample[columns]
print(systematic_sample)


# 2. 整群抽样（假设按某个字段分组抽样）
def cluster_sampling(data, n, cluster_column):
    # 获取所有唯一的群
    clusters = data[cluster_column].unique()

    # 随机选择n个群
    selected_clusters = np.random.choice(clusters, size=min(n, len(clusters)), replace=False)

    # 选择这些群的所有数据
    cluster_sample = data[data[cluster_column].isin(selected_clusters)]

    return cluster_sample

# 按'to_level'列进行整群抽样
data_with_clusters = data_before_sample.copy()
data_with_clusters['to_level'] = data_with_clusters.index // 10  # 每10行一个群

cluster_sample = cluster_sampling(data_with_clusters, 5, 'to_level')  # 使用'to_lvel'列
cluster_sample = cluster_sample[columns]

print("按'to_level'列整群抽样结果:")
print(cluster_sample)






