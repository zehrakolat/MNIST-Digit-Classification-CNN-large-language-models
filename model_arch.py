import torch.nn as nn
import torch.nn.functional as F

class DigitClassifier(nn.Module): 
    def __init__(self):
        super(DigitClassifier, self).__init__()

        # CNN LAYERS
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 28 → 26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 26 → 24

        self.pool = nn.MaxPool2d(2, 2)  # downsample

        # FC LAYERS
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # FIXED SIZE
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN block
        x = self.pool(F.relu(self.conv1(x)))  # 32 x 13 x 13
        x = self.pool(F.relu(self.conv2(x)))  # 64 x 5 x 5

        # flatten
        x = x.view(-1, 64 * 5 * 5)

        # FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x