
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.lstm_cell import *
import common
from model.rnn import *
from model.encorder import *
from model.encoder_senet import *
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
        #self.lstm = SSNet(ENBEDDING_DIM*3, SSNET_HIDDENSIZE, SSNET_OUTPUTSIZE, gpu=self._gpu)
        self.lstm = SSNet_observation_invariance(ENBEDDING_DIM, SSNET_HIDDENSIZE, SSNET_OUTPUTSIZE, gpu=self._gpu)
        self.encoder = encorder(SSNET_OUTPUTSIZE)
        self.encoder_senet = encoder_senet(SSNET_OUTPUTSIZE)

        #wait correct
        # encoder 的输入维度与SPnet不协调
        # SSNET_OUTPUTSIZE

        self.attention = attention(SSNET_OUTPUTSIZE, SSNET_OUTPUTSIZE, label_num)

    def forward(self, input):
        shape = input.shape  # [batch_size, near_num, time_step, feature_dim]
        # query = self.lstm.forward(query_input, SSNET_TIMESTEP)
        #kv, feature_raw = self.encoder.forward(input)
        kv, feature_raw = self.encoder_senet.forward(input)
        feature_raw = feature_raw.permute(0, 2, 3, 1)
        feature_raw = feature_raw[:, 0, :, :]
        flag = input[:, 0, :, 0].unsqueeze(2)
        feature_raw = torch.cat((flag, feature_raw), dim = 2)
        query = self.lstm.forward(feature_raw, SSNET_TIMESTEP)

        output = self.attention.forward(kv, kv, query, input)
        #print(kv[0, :, 62, 0])
        #print(query[0])
        return output

