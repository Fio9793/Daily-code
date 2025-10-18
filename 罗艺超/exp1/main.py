import pandas as pd
import chardet
import random

# 先检测文件编码
with open('data.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

#print(f"检测到的编码: {encoding}")


# 使用检测到的编码读取文件
primitive_data = pd.read_csv("data.csv", encoding=encoding)
primitive_data_1=primitive_data.dropna(how='any')
#print(primitive_data_1)

data_before_filter=primitive_data_1
data_after_filter_1=data_before_filter.loc[data_before_filter["traffic"]!=0]
data_after_filter_2=data_after_filter_1.loc[data_after_filter_1["from_level"]=='一般节点']
#print(data_after_filter_2)

#加权采样
data_before_sample=data_after_filter_2
columns=data_before_sample.columns
weight_sample=data_before_sample.copy()
weight_sample['weight']=0
for i in weight_sample.index:
    if weight_sample.at[i,'to_level']=='一般节点':
        weight=1;
    else :
        weight=5
    weight_sample.at[i,'weight']=weight

weight_sample_finish=weight_sample.sample(n=50,weights='weight')
weight_sample_finish=weight_sample_finish[columns]
#print(weight_sample_finish)

#随机采样
random_sample=data_before_sample
random_sample_finish=random_sample.sample(n=50)
random_sample_finish=random_sample_finish[columns]
#print(random_sample_finish)

#分层抽样
ybjd=data_before_sample.loc[data_before_sample['to_level']=='一般节点']
wlhx=data_before_sample.loc[data_before_sample['to_level']=='网络核心']
after_sample=pd.concat([ybjd.sample(17),wlhx.sample(33)])
#print(after_sample)

#系统抽样，抽50个
n=50
N=len(data_before_sample)
k=N // n #整除返回整数
start=random.randint(0,k-1)
indices=[start +i*k for i in range(n) if (start+i*k)<N]
system_sample=data_before_sample.iloc[indices].reset_index(drop=True)
#print(system_sample)

#整群抽样
unique_clusters=data_before_sample['to_level'].unique()#获取to_level的所有值
select_clusters=random.sample(list(unique_clusters),1)#随机获取一个to_level的群对应的值
print(select_clusters)
cluster_sample=data_before_sample[data_before_sample['to_level'].isin(select_clusters)].reset_index(drop=True)
print(cluster_sample)

