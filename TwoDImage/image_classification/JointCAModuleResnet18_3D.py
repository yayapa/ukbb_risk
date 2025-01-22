import torch
import torch.nn as nn

from TwoDImage.image_classification.LinearClassifierModule import LinearClassifierModule
from TwoDImage.image_classification.ResNet18_3D import ResNet18_3D, ResNet18_3D_Encoder


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, image_features, tabular_features):
        # Ensure tabular features are in the right format for attention (B, 1, D)
        tabular_features = tabular_features.unsqueeze(1)  # Adding sequence dimension
        attn_output, _ = self.attn(image_features.unsqueeze(1), tabular_features, tabular_features)
        return attn_output.squeeze(1)  # Remove sequence dimension

class JointCAModuleResNet18_3D(nn.Module):
    def __init__(
            self,
            num_tabular_features,
            tabular_hidden_layers,
            combine_hidden_layers,
            num_classes=1,
            dropout_rate=0.1,
            n_channels=1,
    ):
        super(JointCAModuleResNet18_3D, self).__init__()

        # MLP for tabular data encoding
        self.tabular_encoder = LinearClassifierModule(
            num_features=num_tabular_features,
            hidden_layers=tabular_hidden_layers,
            dropout_rate=0.2,
            input_dropout_rate=0.2,
        )

        # ResNet18_3D for image encoding
        self.image_encoder = ResNet18_3D_Encoder(
            num_classes=num_classes,
            dropout_prob=dropout_rate,
            num_channels=n_channels
        )

        combined_feature_size = 512

        self.fuse_layer = CrossAttention(embed_dim=combined_feature_size, num_heads=1)

        # MLP for final classification
        self.final_classifier = LinearClassifierModule(
            num_features=combined_feature_size,
            hidden_layers=combine_hidden_layers,
            dropout_rate=0.1,
            num_classes=num_classes
        )

    def forward(self, input):
        tabular_data = input["tabular_vector"]
        image_data = input["image"]
        # Pass tabular data through the tabular MLP
        tabular_features = self.tabular_encoder(tabular_data)

        # Pass image data through the ResNet18_3D
        image_features = self.image_encoder(image_data)

        # Pass through the final MLP
        fused_features = self.fuse_layer(image_features, tabular_features)

        output = self.final_classifier(fused_features)

        return output
