import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    嵌入层，用于得到输入特征指定维度上的嵌入：
        对于连续型特征，通过 nn.Linear 对其进行嵌入
        对于离散型特征，通过 nn.Embedding 对其进行嵌入

    p_count:       病人数量
    n_count:       连续型特征数量
    t_count:       离散型特征字典域大小
    embedded_size: 嵌入的维度
    """
    def __init__(self, n_count, t_count, embedded_size):
        super().__init__()

        self.n_count = n_count
        self.embedded_size = embedded_size

        self.embedding_num_layer = nn.Linear(n_count, n_count * embedded_size)
        self.embedding_text_layer = nn.Embedding(t_count, embedded_size)
        self.activate_fn = nn.LeakyReLU(0.2)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding_num_layer.weight, a=0.2)
        nn.init.kaiming_uniform_(self.embedding_text_layer.weight, a=0.2)

    def forward(self, data):
        """
        num_data: (p_count, n_count)，连续型特征
        text_data: (p_count, t_count)，离散型特征
        """
        num_data, text_data = data  # (p_count, n_count) (p_count, t_count)

        # shape = (p_count, n_count) -> (p_count, n_count * embedded_size) -> (p_count, n_count, embedded_size)
        num_embedded = self.embedding_num_layer(num_data).view(-1, self.n_count, self.embedded_size)

        # shape = (p_count, t_count) -> (p_count, t_count, embedded_size)
        text_embedded = self.embedding_text_layer(text_data)

        # 在 dim = 1 上合并（ dim 从 0 计数）
        # (p_count, (n_count + t_count), embedded_size) -> (p_count, f_count, embedded_size)
        # f_count: 特征总数
        embedded_data = torch.cat((num_embedded, text_embedded), dim=1)
        # 激活函数
        activated_tensor = self.activate_fn(embedded_data)

        return activated_tensor
