# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 16:16:47 2025

@author: 21877
"""

import torch
from torch.utils.data import DataLoader  # 修复导入时的语法错误（缺少空格）
from FCModel import FCModel  # 自定义全连接层模型
from MRPCDataset import MRPCDataset  # 自定义数据集
from transformers import BertTokenizer, BertModel  # 修复逗号分隔问题

# 载入数据
mrpc_dataset = MRPCDataset()  # 变量名规范为小写+下划线（PEP8规范）
train_loader = DataLoader(
    dataset=mrpc_dataset,
    batch_size=16,
    shuffle=True
)
print("数据载入完成")

# 设置运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")  # 更清晰的设备信息输出

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(device)
print("BERT模型加载完成")

# 创建全连接层模型
model = FCModel()
model = model.to(device)
print("全连接层模型创建完成")

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=1e-4)  # 变量名规范
criterion = torch.nn.BCELoss()  # 变量名更清晰（crit -> criterion）

# 计算准确率
def binary_accuracy(predict, label):
    rounded_predict = torch.round(predict)  # 四舍五入得到预测类别（0或1）
    correct = (rounded_predict == label).float()  # 计算正确预测的样本
    accuracy = correct.sum() / len(correct)  # 计算准确率
    return accuracy

# 训练函数
def train():
    # 初始化统计信息
    epoch_loss = 0.0
    epoch_acc = 0.0
    total_len = 0

    # 分batch训练
    for i, data in enumerate(train_loader):
        # 打印当前GPU内存使用（可选，用于调试）
        if torch.cuda.is_available():
            print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # 设置模型为训练模式
        bert_model.train()
        model.train()

        # 取出数据和标签并转移到设备
        sentence, label = data  # 假设MRPCDataset返回(sentence, label)
        label = label.to(device)  # 统一使用to(device)，兼容CPU/GPU

        # 文本编码
        encoding = tokenizer(
         sentence[0],  # 第一个句子
         sentence[1],  # 第二个句子
         return_tensors='pt',
         padding=True,
         truncation=True,
         max_length=128
        )
        # 将编码结果转移到设备
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        # BERT模型前向传播
        with torch.no_grad():  # 若冻结BERT，可添加此行（视需求而定）
            bert_output = bert_model(** encoding)
        pooler_output = bert_output.pooler_output  # 获取[CLS]位置的池化输出

        # 全连接层预测
        predict = model(pooler_output).squeeze()  # 挤压维度匹配标签形状

        # 计算损失和准确率
        loss = criterion(predict, label.float())  # BCELoss要求标签为float类型
        acc = binary_accuracy(predict, label)

        # 反向传播与参数更新
        optimizer.zero_grad()  # 重置梯度
        bert_optimizer.zero_grad()
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新全连接层参数
        bert_optimizer.step()  # 更新BERT参数

        # 累计损失和准确率
        epoch_loss += loss.item() * len(label)  # 使用item()避免计算图累积
        epoch_acc += acc.item() * len(label)
        total_len += len(label)

        # 打印batch信息
        print(f"Batch {i+1} | Loss: {loss.item():.4f} | Accuracy: {acc.item():.4f}")

    # 计算平均损失和准确率
    avg_loss = epoch_loss / total_len
    avg_acc = epoch_acc / total_len
    return avg_loss, avg_acc

# 开始训练
num_epochs = 10  # 变量名规范
for epoch in range(num_epochs):
    epoch_loss, epoch_acc = train()
    print(f"\nEpoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}\n")
