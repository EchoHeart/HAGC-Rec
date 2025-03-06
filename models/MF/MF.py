import torch
import torch.nn as nn

from models.Downstream.MLP import MLP


class MF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, mlp_config):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.MLP = MLP(
            mlp_config['input_features'],
            mlp_config['hidden_features'],
        )

        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, indexes):
        u_g_embeddings = self.user_embedding.weight
        i_g_embeddings = self.item_embedding.weight

        u = u_g_embeddings.index_select(dim=0, index=indexes[0])
        i = i_g_embeddings.index_select(dim=0, index=indexes[1])

        return self.MLP(torch.cat((u, i), dim=1))
