import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout_prob=0.5):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout3d(p=dropout_prob)

        # Fully connected layers are not initialized yet
        self.fc1 = None
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output_size(self, x):
        # Forward pass through conv layers to get output size
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        # Ensure x is on the same device as the model
        x = x.to(next(self.parameters()).device)

        # Dynamically calculate fc1 input size
        if self.fc1 is None:
            conv_out_size = self._get_conv_output_size(x)
            self.fc1 = nn.Linear(conv_out_size, 128)
            self.fc1.apply(initialize_weights)  # Initialize weights for the new layer
            self.fc1 = self.fc1.to(
                next(self.parameters()).device
            )  # Move fc1 to the same device

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def initialize_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def Simple3DCNN_Model(in_channels=1, num_classes=1, dropout_prob=0.5):
    model = Simple3DCNN(in_channels, num_classes, dropout_prob)
    model.apply(initialize_weights)
    return model
