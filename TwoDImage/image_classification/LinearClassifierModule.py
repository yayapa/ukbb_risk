import torch.nn as nn


class LinearClassifierModule(nn.Module):
    def __init__(self, num_features, hidden_layers, dropout_rate=0.1, num_classes=None, input_dropout_rate=None):
        super(LinearClassifierModule, self).__init__()

        layers = []
        input_size = num_features

        for hidden_units in hidden_layers:
            if input_dropout_rate is not None:
                layers.append(nn.Dropout(input_dropout_rate))
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_units

        if num_classes:
            layers.append(nn.Linear(input_size, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
