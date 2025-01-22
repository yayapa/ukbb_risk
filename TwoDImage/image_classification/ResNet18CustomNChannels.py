import torch
import torch.nn as nn
import torchvision.models as models
from TwoDImage.image_classification.LinearClassifierModule import LinearClassifierModule



class ResNet18CustomNChannels(models.ResNet):
    def __init__(self, n_classes=1000, dropout_rate=0.5, n_channels=1, include_fc=True):
        # Load the basic ResNet18 configuration from torchvision
        super().__init__(
            block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=n_classes
        )

        # Adjust the first convolutional layer to accept 7 input channels
        self.conv1 = nn.Conv2d(
            n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize weights for the new convolutional layer
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.include_fc = include_fc
        if not include_fc:
            self.fc = nn.Identity()

        self.add_dropout(dropout_rate)

    def add_dropout(self, dropout_rate):
        # Collect all ReLU activations to a list before modifying the model
        relu_layers = []
        for name, module in self.named_modules():
            if isinstance(module, nn.ReLU):
                relu_layers.append((name, module))

        # Add dropout after each ReLU activation
        for name, module in relu_layers:
            parent_name = name.rsplit(".", 1)[0]
            if parent_name:  # ensure it's not the top level module
                parent = dict(self.named_modules())[parent_name]
                dropout_layer = nn.Dropout(p=dropout_rate, inplace=True)
                # Name the dropout layer based on the ReLU layer's name
                dropout_name = name.split(".")[-1] + "_dropout"
                # Insert the dropout layer after the ReLU layer in the parent module
                parent.add_module(dropout_name, dropout_layer)


class DualEncoderResNet18(nn.Module):
    def __init__(self, n_classes=1, dropout_rate=0.5, n_channels=1, include_fc=False):
        super(DualEncoderResNet18, self).__init__()

        # Create two separate encoders for liver and pancreas
        self.encoder_1 = ResNet18CustomNChannels(dropout_rate=dropout_rate, n_channels=n_channels,
                                                     include_fc=include_fc)
        self.encoder_2 = ResNet18CustomNChannels(dropout_rate=dropout_rate, n_channels=n_channels,
                                                        include_fc=include_fc)

        # Determine the input size for the final FC layer
        if not include_fc:
            # If the FC layers are excluded, the output size will be 512 * 2
            fc_input_size = 1024
        # Fully connected layer for binary classification
            self.fc = nn.Linear(fc_input_size, n_classes)

    def forward(self, input_im):
        input_1 = input_im[0]
        input_2 = input_im[1]
        # Pass each input through its respective encoder
        encoding_1 = self.encoder_1(input_1)
        encoding_2 = self.encoder_2(input_2)

        # Flatten the outputs from both encoders
        encoding_1 = torch.flatten(encoding_1, 1)
        encoding_2 = torch.flatten(encoding_2, 1)

        # Concatenate the outputs from both encoders
        combined_encoding = torch.cat((encoding_1, encoding_2), dim=1)

        # Pass the combined encoding through the final fully connected layer
        output = self.fc(combined_encoding)

        return output

class JointDualEncoderResNet18(nn.Module):
    def __init__(
            self,
            num_tabular_features,
            tabular_hidden_layers,
            combine_hidden_layers,
            num_classes=1,
            dropout_rate=0.1,
            n_channels=1,
            include_fc=False,
    ):
        super(JointDualEncoderResNet18, self).__init__()

        self.encoder_1 = ResNet18CustomNChannels(dropout_rate=dropout_rate, n_channels=n_channels,
                                                     include_fc=include_fc)
        self.encoder_2 = ResNet18CustomNChannels(dropout_rate=dropout_rate, n_channels=n_channels,
                                                        include_fc=include_fc)

        # MLP for tabular data encoding
        self.tabular_encoder = LinearClassifierModule(
            num_features=num_tabular_features,
            hidden_layers=tabular_hidden_layers,
            dropout_rate=dropout_rate,
        )

        num_ftrs = 1024
        combined_feature_size = num_ftrs + tabular_hidden_layers[-1]

        self.final_classifier = LinearClassifierModule(
            num_features=combined_feature_size,
            hidden_layers=combine_hidden_layers,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )

    def forward(self, input):
        tabular_data = input["tabular_vector"]
        tabular_features = self.tabular_encoder(tabular_data)

        image_data = input["image"]
        input_1 = image_data[0]
        input_2 = image_data[1]
        # Pass each input through its respective encoder
        encoding_1 = self.encoder_1(input_1)
        encoding_2 = self.encoder_2(input_2)

        # Flatten the outputs from both encoders
        encoding_1 = torch.flatten(encoding_1, 1)
        encoding_2 = torch.flatten(encoding_2, 1)

        # Pass image data through the ResNet18_3D
        image_features = torch.cat((encoding_1, encoding_2), dim=1)

        # Concatenate the features
        combined_features = torch.cat((tabular_features, image_features), dim=1)

        # Pass through the final MLP
        output = self.final_classifier(combined_features)

        return output

