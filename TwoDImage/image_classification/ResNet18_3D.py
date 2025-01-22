import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.5):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.dropout = nn.Dropout3d(p=dropout_prob)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class ResNet3D(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=1,
        dropout_prob=0.5,
        num_channels=1,
        include_fc=False,
    ):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.dropout = nn.Dropout3d(p=dropout_prob)
        self.layer1 = self._make_layer(
            block, 64, num_blocks[0], stride=2, dropout_prob=dropout_prob
        )
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, dropout_prob=dropout_prob
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, dropout_prob=dropout_prob
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, dropout_prob=dropout_prob
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.include_fc = include_fc
        if include_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_prob))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if self.include_fc:
            out = self.fc(out)
        return out


def initialize_weights(m):
    if isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def ResNet18_3D(num_classes=1, dropout_prob=0.5, num_channels=1):
    model = ResNet3D(
        BasicBlock3D,
        [2, 2, 2, 2],
        num_classes,
        dropout_prob,
        num_channels,
        include_fc=True,
    )
    model.apply(initialize_weights)
    return model


def ResNet18_3D_Encoder(num_classes=1, dropout_prob=0.5, num_channels=50):
    model = ResNet3D(
        BasicBlock3D,
        [2, 2, 2, 2],
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        num_channels=num_channels,
        include_fc=False,
    )
    model.apply(initialize_weights)
    return model


class ResNet18TwoEncoders(nn.Module):
    def __init__(
        self,
        num_classes=1,
        dropout_prob=0.5,
        num_channels=50,
        sax_num_slices=6,
        lax_num_slices=3,
    ):
        super(ResNet18TwoEncoders, self).__init__()
        self.sax_encoder = ResNet18_3D_Encoder(
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            num_channels=num_channels,
        )
        self.lax_encoder = ResNet18_3D_Encoder(
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            num_channels=num_channels,
        )
        self.fc = nn.Linear(1024, num_classes)  # 512 from each encoder
        self.sax_num_slices = sax_num_slices
        self.lax_num_slices = lax_num_slices

    def forward(self, input_im):
        sax_features = self.sax_encoder(input_im[:, :, : self.sax_num_slices, :, :])
        lax_features = self.lax_encoder(input_im[:, :, self.sax_num_slices :, :, :])
        combined_features = torch.cat((sax_features, lax_features), dim=1)
        out = self.fc(combined_features)
        return out


def ResNet18TwoEncoders_3D(
    num_classes=1, dropout_prob=0.5, num_channels=1, sax_num_slices=6, lax_num_slices=3
):
    model = ResNet18TwoEncoders(
        num_classes, dropout_prob, num_channels, sax_num_slices, lax_num_slices
    )
    model.apply(initialize_weights)
    return model
