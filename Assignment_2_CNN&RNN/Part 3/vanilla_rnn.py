from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):

    def __init__(self, input_length, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()
        self.hidden_dim = hidden_dim

        # 初始化权重矩阵和偏置
        self.W_hx = nn.Linear(input_dim, hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        self.W_ph = nn.Linear(hidden_dim, output_dim)
        self.b_o = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(x.size(1)):
            x_t = x[:, t, :]
            h_t = torch.tanh(self.W_hx(x_t) + self.W_hh(h_t) + self.b_h)

        o_t = self.W_ph(h_t) + self.b_o
        y_t = F.log_softmax(o_t, dim=1)
        return y_t