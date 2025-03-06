import torch
import torch.nn as nn


class SoftmaxLayer(nn.Module):
    """
    softmax层

    all_dims: List[int], 特征维度上所有特征的集合
    softmax_dims: List[int], 特征维度上需要进行softmax的特征的集合
    """
    def __init__(self, all_dims, softmax_dims):
        super().__init__()

        self.softmax_dims = softmax_dims
        # 不需要进行softmax的特征的集合由二者的差值得到
        self.no_softmax_dims = list(set(all_dims) - set(softmax_dims))

    def forward(self, data):
        data[:, self.softmax_dims, :] = torch.softmax(data[:, self.softmax_dims, :], dim=1)
        # 不需要进行softmax的特征置0
        data[:, self.no_softmax_dims, :] = 0

        return data
