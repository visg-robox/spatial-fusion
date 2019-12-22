
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.lstm_cell import *
import common
from model.rnn import *
from model.encorder import *
from model.mechanism_attn import *


"""
q_input_size: 生成query的input data维度
kv_
"""
SSNET_HIDDENSIZE = common.qk_dim
SSNET_OUTPUTSIZE = common.qk_dim
SSNET_TIMESTEP = common.time_step
ENBEDDING_DIM = IMG_FEATURE_DIM


class SPNet(nn.Module):
    def __init__(self, input_size, kv_input_size, label_num, gpu=True):
        super(SPNet, self).__init__()
        self._gpu = gpu
        self.lstm = SSNet(ENBEDDING_DIM, SSNET_HIDDENSIZE, SSNET_OUTPUTSIZE, gpu=self._gpu)
        self.encoder = encorder(SSNET_OUTPUTSIZE)

        #wait correct
        # encoder 的输入维度与SPnet不协调
        # SSNET_OUTPUTSIZE

        self.attention = attention(SSNET_OUTPUTSIZE, SSNET_OUTPUTSIZE, label_num)

    def forward(self, input):
        # query = self.lstm.forward(query_input, SSNET_TIMESTEP)
        # kv, _ = self.encoder.forward(input)
        img_feature_raw = input[:, :, :, :1+common.feature_num_i]
        shape = img_feature_raw.shape  # [batch_size, near_num, time_step, feature_dim]

        # feature_raw = feature_raw[:, 0, :, :]
        img_feature_raw = img_feature_raw.view(shape[0]*shape[1], shape[2], shape[3])
        # feature_raw = torch.cat((flag, feature_raw), dim = 2)
        kv = self.lstm.forward(img_feature_raw, SSNET_TIMESTEP)
        kv = kv.view(shape[0], shape[1], kv.shape[1])
        kv = kv.unsqueeze(2)
        q = kv[:, 0, 0, :]

        output = self.attention.forward(kv, kv, q, input)
        #print(kv[0, :, 62, 0])
        #print(query[0])
        return output

