# FCModel.py
import torch
import torch.nn as nn

class FCModel(nn.Module):
    def __init__(self, hidden_dim=768, dropout=0.3):
        super(FCModel, self).__init__()
        # 全连接层：将BERT的768维输出映射到1维
        self.fc1 = nn.Linear(hidden_dim, 256)  # 中间层（可选，增加非线性能力）
        self.fc2 = nn.Linear(256, 1)           # 输出层（单值概率）
        self.relu = nn.ReLU()                  # 激活函数
        self.dropout = nn.Dropout(dropout)     # 防止过拟合
        self.sigmoid = nn.Sigmoid()            # 输出概率（0-1）

    def forward(self, x):
        # x: 输入形状 (batch_size, 768)，即BERT的pooler_output
        x = self.fc1(x)       # (batch_size, 256)
        x = self.relu(x)      # 激活
        x = self.dropout(x)   #  dropout
        x = self.fc2(x)       # (batch_size, 1)
        x = self.sigmoid(x)   # 输出概率
        return x