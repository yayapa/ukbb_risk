import torch.nn as nn
import torch
from TwoDImage.image_classification.LinearClassifierModule import LinearClassifierModule
from TwoDImage.image_classification.ResNet18Custom import ResNet18Custom


class JointModule(nn.Module):
    def __init__(
        self,
        num_tabular_features,
        tabular_hidden_layers,
        combine_hidden_layers,
        num_classes=1,
        dropout_rate=0.1,
    ):
        super(JointModule, self).__init__()

        # ResNet18 for image data
        self.image_module = ResNet18Custom(n_classes=num_classes)
        num_ftrs = self.image_module.fc.in_features
        self.image_module.fc = nn.Identity()  # Remove the final classification layer

        # Tabular model for tabular data
        self.tabular_module = LinearClassifierModule(
            num_features=num_tabular_features,
            hidden_layers=tabular_hidden_layers,
            num_classes=None,
            dropout_rate=dropout_rate,
        )

        # Combined MLP
        combined_input_size = num_ftrs + tabular_hidden_layers[-1]
        self.combine_module = LinearClassifierModule(
            num_features=combined_input_size,
            hidden_layers=combine_hidden_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    def forward(self, input):
        # Forward pass through ResNet18
        image_features = self.image_module(input["image"])

        # Forward pass through TabularModel
        tabular_features = self.tabular_module(input["tabular_vector"])

        # Concatenate image and tabular features
        combined_features = torch.cat((image_features, tabular_features), dim=1)

        # Forward pass through combined MLP
        output = self.combine_module(combined_features)

        return output
