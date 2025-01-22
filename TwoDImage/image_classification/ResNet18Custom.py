import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Custom(models.ResNet):
    def __init__(self, n_classes=1000, dropout_rate=0.5):
        # Load the basic ResNet18 configuration from torchvision
        super().__init__(
            block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=n_classes
        )
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
