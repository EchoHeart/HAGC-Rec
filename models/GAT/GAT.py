import torch.nn as nn
from models.GAT.GATLayer import GATLayer


class GAT(nn.Module):
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, dropout_prob, bias,
                 add_skip_connection):
        super().__init__()

        # 输入视为1个注意力头
        num_heads_per_layer = [1] + num_heads_per_layer

        GAT_layers = []
        for k in range(num_of_layers):
            GAT_layer = GATLayer(
                num_in_features=num_heads_per_layer[k] * num_features_per_layer[k],
                num_of_heads=num_heads_per_layer[k + 1],
                num_out_features=num_features_per_layer[k + 1],
                dropout_prob=dropout_prob,
                activation=nn.ELU() if k < num_of_layers - 1 else None,
                bias=bias,
                concat=True if k < num_of_layers - 1 else False,
                add_skip_connection=add_skip_connection
            )
            GAT_layers.append(GAT_layer)

        self.GAT = nn.Sequential(*GAT_layers)

    def forward(self, data):
        return self.GAT(data)
