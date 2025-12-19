import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from MRPCDataset import MRPCDataset
from FCModel import FCModel
from transformers import BertTokenizer, BertModel
import time
import os

# ！！！ 必须放在最顶部，在所有import之前 ！！！
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class MRPCTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU信息: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        self.setup_directories()
        self.load_data()
        self.setup_models()
        self.setup_optimizers()

    def setup_directories(self):
        """创建必要的目录"""
        os.makedirs('models', exist_ok=True)  # 用于保存模型
        os.makedirs('data', exist_ok=True)    # 用于缓存数据集
        print("创建目录: 'models', 'data'")

    def load_data(self):
        """加载数据"""
        print("\n" + "=" * 50)
        print("加载数据集...")

        self.train_dataset = MRPCDataset(split='train')
        self.val_dataset = MRPCDataset(split='validation')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )

    def setup_models(self):
        """设置模型"""
        print("\n初始化模型...")

        # 初始化tokenizer和BERT (会自动从镜像站下载)
        print("正在下载/加载 BERT tokenizer 和模型...")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.to(self.device)
        print("BERT模型加载完成！")

        # 创建分类头
        self.classifier = FCModel()
        self.classifier.to(self.device)
        print("全连接分类头创建完成。")

    def setup_optimizers(self):
        """设置优化器"""
        # BERT部分使用较小的学习率
        self.bert_optimizer = torch.optim.AdamW(
            self.bert_model.parameters(),
            lr=self.config['bert_lr']
        )

        # 分类头使用较大的学习率
        self.classifier_optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.config['classifier_lr']
        )

        # 损失函数
        self.criterion = nn.BCELoss()

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.bert_optimizer,
            step_size=2,
            gamma=0.1
        )

    def binary_accuracy(self, preds, labels):
        """计算准确率"""
        rounded_preds = torch.round(preds)
        correct = (rounded_preds == labels).float()
        acc = correct.sum() / len(correct)
        return acc

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.bert_model.train()
        self.classifier.train()

        epoch_loss = 0
        epoch_acc = 0
        total_samples = 0

        for batch_idx, (sentences, labels) in enumerate(self.train_loader):
            # 将数据移到设备
            labels = labels.to(self.device)

            # Tokenization
            encoding = self.tokenizer(
                sentences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )

            # 移到设备
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # 前向传播
            bert_output = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 使用[CLS] token的表示
            cls_output = bert_output.last_hidden_state[:, 0, :]

            # 分类
            predictions = self.classifier(cls_output).squeeze()

            # 计算损失
            loss = self.criterion(predictions, labels)

            # 反向传播
            self.bert_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)

            # 更新参数
            self.bert_optimizer.step()
            self.classifier_optimizer.step()

            # 计算准确率
            acc = self.binary_accuracy(predictions, labels)

            # 统计
            batch_size = len(labels)
            epoch_loss += loss.item() * batch_size
            epoch_acc += acc.item() * batch_size
            total_samples += batch_size

            # 每10个batch打印一次
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config['epochs']} | "
                      f"Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {acc.item():.4f}")

        # 计算平均损失和准确率
        avg_loss = epoch_loss / total_samples
        avg_acc = epoch_acc / total_samples

        return avg_loss, avg_acc

    def validate(self):
        """验证模型"""
        self.bert_model.eval()
        self.classifier.eval()

        val_loss = 0
        val_acc = 0
        total_samples = 0

        with torch.no_grad():
            for sentences, labels in self.val_loader:
                labels = labels.to(self.device)

                encoding = self.tokenizer(
                    sentences,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                bert_output = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                cls_output = bert_output.last_hidden_state[:, 0, :]
                predictions = self.classifier(cls_output).squeeze()

                loss = self.criterion(predictions, labels)
                acc = self.binary_accuracy(predictions, labels)

                batch_size = len(labels)
                val_loss += loss.item() * batch_size
                val_acc += acc.item() * batch_size
                total_samples += batch_size

        avg_loss = val_loss / total_samples
        avg_acc = val_acc / total_samples

        return avg_loss, avg_acc

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 50)
        print("开始训练...")
        print("=" * 50)

        best_val_acc = 0

        for epoch in range(self.config['epochs']):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate()

            # 更新学习率
            self.scheduler.step()

            # 打印epoch结果
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch + 1} 完成:")
            print(f"  时间: {epoch_time:.2f}秒")
            print(f"  训练 Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  验证 Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print("-" * 50)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_save_path = f"models/best_model_epoch{epoch + 1}.pth"
                self.save_model(model_save_path)
                print(f"  保存最佳模型至: {model_save_path}")

        print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")

    def save_model(self, path):
        """保存模型 (仅保存状态字典)"""
        torch.save({
            'bert_state_dict': self.bert_model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'config': self.config
        }, path)

    def test_example(self):
        """测试示例"""
        print("\n" + "=" * 50)
        print("测试示例句子对:")

        examples = [
            ("The cat sits on the mat", "The cat is on the mat", 1),
            ("I love programming", "Coding is enjoyable", 1),
            ("The weather is nice", "It's raining outside", 0),
        ]

        self.bert_model.eval()
        self.classifier.eval()

        with torch.no_grad():
            for s1, s2, true_label in examples:
                sentence = f"{s1} [SEP] {s2}"

                encoding = self.tokenizer(
                    sentence,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                bert_output = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                cls_output = bert_output.last_hidden_state[:, 0, :]
                prediction = self.classifier(cls_output).squeeze()

                pred_label = 1 if prediction.item() > 0.5 else 0
                confidence = prediction.item() if pred_label == 1 else 1 - prediction.item()

                print(f"\n句子1: {s1}")
                print(f"句子2: {s2}")
                print(f"真实标签: {true_label} ({'同义' if true_label == 1 else '不同义'})")
                print(f"预测标签: {pred_label} ({'同义' if pred_label == 1 else '不同义'})")
                print(f"置信度: {confidence:.4f}")
                print(f"是否正确: {'✓' if pred_label == true_label else '✗'}")

def main():
    """主函数"""
    # 配置参数
    config = {
        'batch_size': 16,
        'epochs': 3,  # 根据实验要求可以设置为1
        'bert_lr': 2e-5,
        'classifier_lr': 1e-3,
        'max_length': 128,
    }

    # 创建训练器
    trainer = MRPCTrainer(config)

    # 开始训练
    trainer.train()

    # 测试示例
    trainer.test_example()

if __name__ == "__main__":
    main()