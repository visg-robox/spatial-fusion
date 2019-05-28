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


class Query(nn.Module):
    def __init__(self, input_size, kv_input_size, label_num):
        super(Query, self).__init__()
        self.lstm = SSNet(input_size, SSNET_HIDDENSIZE, SSNET_OUTPUTSIZE, gpu=self._gpu)

    def forward(self, input):
        shape = input.shape  # [batch_size, near_num, time_step, feature_dim]
        image_input = self.tanh(input[:, shape[1] // 2, :, :common.img_feature_size])
        image_input = self.tanh1(image_input)
        vector_input = self.tanh2(
            input[:, shape[1] // 2, :, common.img_feature_size:common.img_feature_size + common.vector_size])

        query = self.lstm.forward(query_input, SSNET_TIMESTEP)
        kv = self.encoder.forward(input)

        output = self.attention.forward(kv, kv, query, input)
        # print(kv[0, :, 62, 0])
        # print(query[0])
        return output

