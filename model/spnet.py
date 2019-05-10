
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.lstm_cell import *
import common
from model.rnn import *
"""
q_input_size: 生成query的input data维度
kv_
"""
SSNET_HIDDENSIZE = 256
SSNET_OUTPUTSIZE = 256
SSNET_TIMESTEP = common.time_step

class SPNet(nn.Module):
    def __init__(self, q_input_size, kv_input_size, output_size, gpu=True):
        super(SPNet, self).__init__()
        self._gpu = gpu
        SSNet(None, SSNET_HIDDENSIZE, SSNET_OUTPUTSIZE, gpu=self._gpu)

    def forward(self, input):
        q_input = 
        kv_input =
        kv = encode(kv_input)
        query = SSNet(q_input, SSNET_TIMESTEP)
        output = attention(query, kv)