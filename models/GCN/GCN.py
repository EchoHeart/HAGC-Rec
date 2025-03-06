import torch.nn as nn
from models.GCN.GCNLayer import GraphConvolution


class GCN(nn.Module):
    def __init__(self, input_features, hidden_features, out_features, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)

        self.Dropout = nn.Dropout(dropout)
        self.ReLU = nn.ReLU()

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.ReLU(x)
        x = self.Dropout(x)
        x = self.gc2(x, adj)
        return x
