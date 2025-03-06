import torch
import torch.nn as nn
from models.HAT.SoftmaxLayer import SoftmaxLayer

import numpy as np


class HATLayer(nn.Module):
    """
    单层HAN

    in_features: 输入的嵌入维度
    out_features: 线性层输出的嵌入维度
    num_softmax_layers: softmax层的数量, num_softmax_layers == len(softmax_dims)
    is_last: 是否为最后一层
    """

    def __init__(self, in_features, out_features, num_softmax_layers, all_dims, softmax_dims, is_last):
        super().__init__()

        self.out_features = out_features
        self.all_dims = all_dims
        self.is_last = is_last

        self.linear_layer = nn.Linear(in_features, out_features, bias=False)
        self.score_layer = nn.Parameter(torch.Tensor(1, out_features))
        self.activate_fn = nn.LeakyReLU(0.2)

        softmax_layers = []
        for k in range(num_softmax_layers):
            softmax_layer = SoftmaxLayer(range(all_dims), softmax_dims[k])
            softmax_layers.append(softmax_layer)
        # 将所有的softmax层并联起来！！！
        self.softmax_layers = nn.ModuleList(softmax_layers)
        self.softmax_dims = softmax_dims

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.linear_layer.weight, a=0.2)
        nn.init.kaiming_uniform_(self.score_layer, a=0.2)

    def forward(self, data):
        """
        data.shape = [p_count, f_count, embedded_size]
        f_count -> all_dims
        embedded_size -> in_features
        """
        # (p_count, all_dims, in_features) -> (p_count, all_dims, out_features)
        lineared_tensor = self.linear_layer(data)

        # element-wise product & sum
        # (p_count, all_dims, out_features) -> (p_count, all_dims, 1)
        scored_tensor = (lineared_tensor * self.score_layer).sum(dim=-1, keepdims=True)
        activated_tensor = self.activate_fn(scored_tensor)  # (p_count, all_dims, 1)

        """
        1. 并联的每个softmax层独立地对输入进行处理
        2. 由于每个层所需要执行softmax的特征集合不同，使得每个层得到的输出都是在给定特征集合上执行softmax的结果
        3. 置0操作确保了softmax结果不受非给定特征集合的影响
        """
        after_softmax_tensors = []
        # 每个softmax层独立地对输入进行处理
        for i, softmax_layer in enumerate(self.softmax_layers):
            softmax_tensor = softmax_layer(activated_tensor)

            # suffix = 'nurse' if softmax_tensor.shape[0] == 200 else 'patient'
            # softmax_tensor_np = softmax_tensor.squeeze().cpu().detach().numpy()[:, self.softmax_dims[i]]
            # np.save(f'{self.all_dims}_{i}_{suffix}.npy', softmax_tensor_np)

            # element-wise product & sum
            # (p_count, all_dims, out_features) * (p_count, all_dims, 1) -> (p_count, all_dims, out_features)
            after_softmax_tensor = lineared_tensor * softmax_tensor.clone()

            # 如果不是最后一层，通过 sum 在特征维度上进行聚合
            if not self.is_last:
                # (p_count, all_dims, out_features) -> (p_count, 1, out_features)
                after_softmax_tensor = after_softmax_tensor.sum(dim=1, keepdims=True)
            # 最后一层进行拼接
            else:
                return after_softmax_tensor.view(-1, self.all_dims * self.out_features)

            after_softmax_tensors.append(after_softmax_tensor)

        return torch.cat(after_softmax_tensors, dim=1)  # (p_count, num_softmax_layers, out_features)
