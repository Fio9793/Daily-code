import torch
from torch.utils.data import Dataset
from datasets import load_dataset  # 直接导入


class MRPCDataset(Dataset):
    def __init__(self, split='train'):
        """
        加载MRPC数据集 (使用HuggingFace datasets库)
        split: 'train', 'validation', 'test'
        """
        super().__init__()
        self.split = split
        print(f"正在从HuggingFace加载 MRPC {split} 集...")

        # 直接加载GLUE基准中的MRPC数据集
        self.dataset = load_dataset('glue', 'mrpc', split=split)
        print(f"成功加载，共 {len(self.dataset)} 个样本。")

        # 验证数据格式 (可选，运行时可注释掉)
        # print("数据示例:", self.dataset[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 将两个句子合并，用[SEP]分隔
        sentence = f"{item['sentence1']} [SEP] {item['sentence2']}"
        label = torch.tensor(item['label'], dtype=torch.float32)
        return sentence, label