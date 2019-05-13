import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


#相关系数的求解使用的是K和Q的点乘

class attention(nn.Module):
    def __init__(self, k_dim, v_dim, label_num):
        super(attention, self).__init__()
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.label_num = label_num

        self.decoder = nn.Linear(self.v_dim, self.label_num)

    def forward(self, K, V, Q, input_data):
        K = K.permute(0, 2, 3, 1)
        V = V.permute(0, 2, 3, 1)
        flag = input_data[:, :, :, 0]
        shape = K.shape
        K = K.view(shape[0], shape[1] * shape[2], shape[3])
        V = V.view(shape[0], shape[1] * shape[2], shape[3])
        flag = flag.view(shape[0], shape[1] * shape[2])

        attn_weights = self.get_attn_weights(K, Q, flag)  # [bz, near_num*time_step, 1]
        attn_value = attn_weights.permute(0, 2, 1).bmm(V)
        output = self.decoder(attn_value.squeeze(1))
        output = F.softmax(output).view(shape[0], self.label_num)

        return output

    def get_attn_weights(self, k, q, flag):
        shape = k.shape  #   k [bz, near_num*time_step, feature]
                         #   q [bz, feature]
                         #   flag [bz, near_num*time_step]

        q = q.unsqueeze(1).expand(-1, shape[1], -1)
        attn_scores = self.get_attn_score(k, q) #[bz, near_num*time_step]

        attn_scores = torch.exp(attn_scores)
        attn_sum = torch.sum(attn_scores*flag, 1).unsqueeze(1).expand(-1, shape[1])
        attn_scores_soft = (attn_scores*flag/attn_sum).view(shape[0], shape[1], 1)

        return attn_scores_soft

    def get_attn_score(self, k, q):
        score = torch.sum(k*q, 2)
        return score