import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes=10, dropout=0.5):
        super(SimpleCNN, self).__init__()

        # Define a simple convolutional neural network
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(
            256 * 36 * 48, 1024
        )  # Adjust according to input dimensions after conv layers
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(
            512, num_classes
        )  # Adjust number of output classes as necessary

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 256 * 36 * 48)  # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
