import torch.nn as nn

from models.Embedding.EmbeddingLayer import EmbeddingLayer
from models.GAT.GAT import GAT
from models.HAT.HAT import HAT
from models.GCN.GCN import GCN


class UserModel(nn.Module):
    def __init__(self,
                 embeddingLayer_config=None,
                 useHAT=False, hat_config=None,
                 useGAT=False, gat_config=None,
                 useGCN=False, gcn_config=None):
        super().__init__()

        self.useHAT = useHAT
        self.useGAT = useGAT
        self.useGCN = useGCN

        # 嵌入层
        self.embeddingLayer = EmbeddingLayer(
            n_count=embeddingLayer_config['n_count'],
            t_count=embeddingLayer_config['t_count'],
            embedded_size=embeddingLayer_config['embedded_size']
        )

        # 建模层
        self.hat = HAT(
            num_HAT_layers=hat_config['num_HAT_layers'],
            in_features=hat_config['in_features'],
            out_features=hat_config['out_features'],
            num_softmax_layers=hat_config['num_softmax_layers'],
            all_dims=hat_config['all_dims'],
            softmax_dims=hat_config['softmax_dims']
        ) if hat_config else None

        self.gat = GAT(
            num_of_layers=gat_config['num_of_layers'],
            num_heads_per_layer=gat_config['num_heads_per_layer'],
            num_features_per_layer=gat_config['num_features_per_layer'],
            dropout_prob=gat_config['dropout_prob'],
            bias=gat_config['bias'],
            add_skip_connection=gat_config['add_skip_connection']
        ) if gat_config else None

        self.gcn = GCN(
            input_features=gcn_config['input_features'],
            hidden_features=gcn_config['hidden_features'],
            out_features=gcn_config['out_features'],
            dropout=gcn_config['dropout']
        ) if gcn_config else None

    def forward(self, data):
        num_data, text_data = data[0:2]
        edge_data = data[2] if self.useGAT else None
        adj_data = data[2] if self.useGCN else None

        after_embeddingLayer_tensor = self.embeddingLayer((num_data, text_data))
        if self.useHAT and self.useGAT:
            after_hat_tensor = self.hat(after_embeddingLayer_tensor)
            after_gat_tensor = self.gat((after_hat_tensor, edge_data))[0]
            return after_gat_tensor

        if self.useHAT and self.useGCN:
            after_hat_tensor = self.hat(after_embeddingLayer_tensor)
            after_gcn_tensor = self.gcn(after_hat_tensor, adj_data)
            return after_gcn_tensor

        elif self.useHAT:
            return self.hat(after_embeddingLayer_tensor)

        elif self.useGAT:
            [f_count, embedded_size] = after_embeddingLayer_tensor.shape[1:]
            return self.gat((after_embeddingLayer_tensor.view(-1, f_count * embedded_size), edge_data))[0]

        elif self.useGCN:
            [f_count, embedded_size] = after_embeddingLayer_tensor.shape[1:]
            return self.gcn(after_embeddingLayer_tensor.view(-1, f_count * embedded_size), adj_data)
