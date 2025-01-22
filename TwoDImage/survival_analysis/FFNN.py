import torch


class FFNN(torch.nn.Module):
    def __init__(self, num_nodes, dropout=0.1):
        super(FFNN, self).__init__()
        layers = []
        for i in range(len(num_nodes) - 2):
            layers.append(torch.nn.Linear(num_nodes[i], num_nodes[i + 1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(num_nodes[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(num_nodes[-2], num_nodes[-1]))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
