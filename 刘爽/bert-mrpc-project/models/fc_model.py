import torch.nn as nn
from config import config

class FCModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=1):
        super(FCModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(x)