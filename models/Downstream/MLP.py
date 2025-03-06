import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, num_i, num_h, num_o=1):
        super().__init__()

        self.linear1 = nn.Linear(num_i, num_h)
        self.LR = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.LR(x)
        x = self.linear2(x)
        return x
