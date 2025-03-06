import torch
import torch.nn as nn
from models.HAT.HATLayer import HATLayer


class HAT(nn.Module):
    def __init__(self, num_HAT_layers, in_features, out_features, num_softmax_layers, all_dims, softmax_dims):
        super().__init__()

        HAT_layers = []
        for k in range(num_HAT_layers):
            layer = HATLayer(
                in_features=in_features[k],
                out_features=out_features[k],
                num_softmax_layers=num_softmax_layers[k],
                all_dims=all_dims[k],
                softmax_dims=softmax_dims[k],
                is_last=True if k == num_HAT_layers - 1 else False
            )
            HAT_layers.append(layer)

        self.HAT = nn.Sequential(*HAT_layers)

    def forward(self, data):
        return self.HAT(data)
