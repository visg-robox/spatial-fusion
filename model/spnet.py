
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

class SPNet(nn.Module):
    def __init__(self, input_size, kv_input_size, label_num, gpu=True):
        super(SPNet, self).__init__()
        self._gpu = gpu
        self.lstm = SSNet(input_size, SSNET_HIDDENSIZE, SSNET_OUTPUTSIZE, gpu=self._gpu)
        self.encoder = encorder(SSNET_OUTPUTSIZE)
        #wait correct
        # encoder 的输入维度与SPnet不协调
        # SSNET_OUTPUTSIZE
        self.decoder = nn.Conv2d(256, 13, 1)
        self.attention = attention(SSNET_OUTPUTSIZE, SSNET_OUTPUTSIZE, label_num)

    def forward(self, input):
        shape = input.shape  # [batch_size, near_num, time_step, feature_dim]
        query = self.lstm.forward(input[:, shape[1]//2, :, :], SSNET_TIMESTEP)
        kv = self.encoder.forward(input)


        # output = self.attention.forward(kv, kv, query, input)
        out = kv[:,:,62:63,0:1]
        out = self.decoder(out)
        out = out.squeeze()
        return out