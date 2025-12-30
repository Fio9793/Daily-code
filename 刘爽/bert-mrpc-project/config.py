import torch

class Config:
    # 数据配置
    data_path = "./data/mrpc"
    batch_size = 16
    max_length = 128
    
    # 模型配置
    model_name = "bert-base-uncased"
    hidden_size = 768
    num_labels = 2
    
    # 训练配置
    learning_rate = 2e-5
    bert_learning_rate = 2e-5
    num_epochs = 3
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()