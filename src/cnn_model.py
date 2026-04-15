import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainTumorCNN(nn.Module):
    """5-Layer CNN as described in Paper 4"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)

        # Calculate flat size: input 128x128 → after 3 pools → 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)   # 2 classes: tumor / no tumor

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 64x64
        x = self.pool(F.relu(self.conv2(x)))   # 32x32
        x = self.pool(F.relu(self.conv3(x)))   # 16x16
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)