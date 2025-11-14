# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class MRPCDataset(Dataset):
    def __init__(self, data_dir=None, split='train'):
        """
        加载MRPC txt格式数据集
        :param data_dir: 数据集存放目录（默认与脚本同目录）
        :param split: 数据集类型（'train'或'test'）
        """
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 匹配txt文件名
        file_name = f"msr_paraphrase_{split}.txt"
        self.file_path = os.path.join(data_dir, file_name)
        
        # 读取txt数据（MRPC txt格式为：标签\t句子1\t句子2）
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过表头行
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    label = int(parts[0])
                    sent1 = parts[1]
                    sent2 = parts[2]
                    data.append({'Quality': label, '#1 String': sent1, '#2 String': sent2})
        
        self.data = pd.DataFrame(data)
        
        # 检查必要列
        required_columns = ['Quality', '#1 String', '#2 String']
        if not set(required_columns).issubset(self.data.columns):
            raise ValueError(f"数据集文件缺少必要列，请确保包含：{required_columns}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        label = item['Quality']
        sentence1 = item['#1 String']
        sentence2 = item['#2 String']
        return (sentence1, sentence2), label
