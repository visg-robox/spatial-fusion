import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import common

#相关系数的求解使用的是K和Q的点乘

class attention(nn.Module):
    def __init__(self, k_dim, v_dim, label_num, shape = (common.near_num, 1)):
        super(attention, self).__init__()
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.label_num = label_num
        self.maxpool_2d = nn.AdaptiveMaxPool2d(shape)
        self.decoder = nn.Linear(self.v_dim, self.label_num)
        #self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, K0, V0, Q, input_data):
        K = K0
        V = V0
        flag = input_data[:, :, :, 0:1]
        flag = flag.permute(0, 3, 1, 2)
        flag = self.maxpool_2d(flag)

        shape = K.shape
        K = K.view(shape[0], shape[1] * shape[2], shape[3])
        V = V.view(shape[0], shape[1] * shape[2], shape[3])
        flag = flag.view(shape[0], shape[1] * shape[2])

        attn_weights = self.get_attn_weights(K, Q, flag)  # [bz, near_num*time_step, 1]
        attn_value = attn_weights.permute(0, 2, 1).bmm(V)
        output = self.decoder(attn_value.squeeze(1))
        output = output.view(shape[0], self.label_num)

        return output

    def get_attn_weights(self, k, q, flag):
        shape = k.shape  #   k [bz, near_num*time_step, feature]
                         #   q [bz, feature]
                         #   flag [bz, near_num*time_step]

        q = q.unsqueeze(1).expand(-1, shape[1], -1)
        attn_scores = self.get_attn_score(k, q) #[bz, near_num*time_step]

        attn_scores = torch.exp(attn_scores)
        attn_sum = torch.sum(attn_scores*flag, 1).unsqueeze(1).expand(-1, shape[1])
        attn_scores_soft = (attn_scores*flag/(attn_sum+1e-5)).view(shape[0], shape[1], 1)

        return attn_scores_soft

    def get_attn_score(self, k, q):
        score = torch.sum(k*q, 2)/np.power(k.shape[2], 0.5)
        score = self.dropout(score)
        return score