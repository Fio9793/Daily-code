# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 16:29:12 2025

@author: 21877
"""
# FCModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=1):
        """
        全连接层模型，用于BERT输出的分类
        :param input_dim: BERT输出的特征维度（bert-base-uncased默认768）
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出维度（MRPC是二分类，输出1个概率值）
        """
        super(FCModel, self).__init__()
        # 定义全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # 激活函数和 dropout（防止过拟合）
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        """
        前向传播
        :param x: BERT输出的[CLS]池化特征，形状为 (batch_size, input_dim)
        :return: 二分类概率（经过sigmoid）
        """
        x = self.fc1(x)       # 第一层全连接
        x = self.relu(x)      # 激活函数
        x = self.dropout(x)   # dropout层
        x = self.fc2(x)       # 第二层全连接（输出1维）
        x = torch.sigmoid(x)  # 转换为概率（0~1之间）
        return x

