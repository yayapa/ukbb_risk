import torch
import torch.nn as nn

from TwoDImage.image_classification.LinearClassifierModule import LinearClassifierModule
from TwoDImage.image_classification.ResNet18_3D import ResNet18_3D


class JointModuleResNet18_3D(nn.Module):
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
        super(JointModuleResNet18_3D, self).__init__()

        # MLP for tabular data encoding
        self.tabular_encoder = LinearClassifierModule(
            num_features=num_tabular_features,
            hidden_layers=tabular_hidden_layers,
            dropout_rate=tabular_dropout_rate,
            input_dropout_rate=tabular_input_dropout_rate,
            #dropout_rate=dropout_rate,
        )

        # ResNet18_3D for image encoding
        self.image_encoder = ResNet18_3D(
            num_classes=num_classes,
            dropout_prob=dropout_rate,
            num_channels=n_channels
        )

        if restore_image_model_path is not None:
            print("Loaded pre-trained model from: ", restore_image_model_path)
            self.image_encoder.load_state_dict(torch.load(restore_image_model_path))

        num_ftrs = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()  # Remove the final classification layer

        # Combine encoded features from tabular and image data
        combined_feature_size = tabular_hidden_layers[-1] + num_ftrs  # 512 is the final output size of ResNet18_3D

        # MLP for final classification
        self.final_classifier = LinearClassifierModule(
            num_features=combined_feature_size,
            hidden_layers=combine_hidden_layers,
            dropout_rate=combine_dropout_rate,
            num_classes=num_classes
        )

    def forward(self, input):
        tabular_data = input["tabular_vector"]
        image_data = input["image"]
        # Pass tabular data through the tabular MLP
        tabular_features = self.tabular_encoder(tabular_data)

        # Pass image data through the ResNet18_3D
        image_features = self.image_encoder(image_data)

        # Concatenate the features
        combined_features = torch.cat((tabular_features, image_features), dim=1)

        # Pass through the final MLP
        output = self.final_classifier(combined_features)

        return output
