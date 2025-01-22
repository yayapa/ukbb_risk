import torch.nn as nn
import torch
from TwoDImage.image_classification.LinearClassifierModule import LinearClassifierModule
from TwoDImage.image_classification.ResNet18CustomNChannels import (
    ResNet18CustomNChannels,
)


class JointModuleResNet18NChannels(nn.Module):
    def __init__(
        self,
        num_tabular_features,
        tabular_hidden_layers,
        combine_hidden_layers,
        tabular_dropout_rate=0.0,
        tabular_input_dropout_rate=0.0,
        combine_dropout_rate=0.0,
        num_classes=1,
        dropout_rate=0.1,
        n_channels=1,
        restore_image_model_path=None
    ):
        super(JointModuleResNet18NChannels, self).__init__()

        # ResNet18 for image data
        self.image_module = ResNet18CustomNChannels(
            n_classes=num_classes,
            dropout_rate=dropout_rate,
            n_channels=n_channels
        )

        if restore_image_model_path is not None:
            print("Loaded pre-trained model from: ", restore_image_model_path)
            self.image_module.load_state_dict(torch.load(restore_image_model_path))

        num_ftrs = self.image_module.fc.in_features
        self.image_module.fc = nn.Identity()  # Remove the final classification layer

        # Tabular model for tabular data
        self.tabular_module = LinearClassifierModule(
            num_features=num_tabular_features,
            hidden_layers=tabular_hidden_layers,
            num_classes=None,
            dropout_rate=tabular_dropout_rate,
            input_dropout_rate=tabular_input_dropout_rate,
        )

        # Combined MLP
        combined_input_size = num_ftrs + tabular_hidden_layers[-1]
        self.combine_module = LinearClassifierModule(
            num_features=combined_input_size,
            hidden_layers=combine_hidden_layers,
            num_classes=num_classes,
            dropout_rate=combine_dropout_rate,
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
