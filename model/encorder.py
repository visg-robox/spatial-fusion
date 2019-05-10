# Encorder for voxel_feature to  encoded feature
#
# input: (batch_size, near_num, time_step, 1(flag) + img_feature_dim + vector_dim + offset_dim)
# output:(batch_size, near_num, time_step, qk_dim)
#
# model:
# vector,offset --> expand to same dim as img_feature_dim
# concat
# MLP

#TODO(luo) : completed, TRANSPOSE, SLICE
#TODO(luo) : completed, MLP, relu, bias,　由于有很多无效点，所以考虑不使用ＢＮ层


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.lstm_cell import *
import common
from model.rnn import *

IMG_FEATURE_DIM = 128
VECTOR_DIM = 3
VECTOR_EN_DIM = IMG_FEATURE_DIM
OFFSET_DIM = 3
OFFSET_EN_DIM = IMG_FEATURE_DIM


class encorder(nn.Module):
    def __init__(self, gpu=True):
        super(encorder, self).__init__()
        self._gpu = gpu
        self.conv_vector = nn.Conv2d(VECTOR_DIM, VECTOR_EN_DIM, 1)
        self.conv_offset = nn.Conv2d(OFFSET_DIM, OFFSET_EN_DIM, 1)
        self.conv1 = nn.Conv2d(IMG_FEATURE_DIM + VECTOR_EN_DIM + OFFSET_EN_DIM, common.qk_dim, 1)
        self.conv2 = nn.Conv2d(common.qk_dim, common.qk_dim, 1)




    def forward(self, input):
        transpose_input = input.permute(0,3,1,2)
        flag = transpose_input[:,0,:,:]
        img_feature = transpose_input[:,1 : 1 + IMG_FEATURE_DIM,:,:]
        vector = transpose_input[:,1 + IMG_FEATURE_DIM :1 + IMG_FEATURE_DIM + VECTOR_DIM,:,:]
        offset = transpose_input[:,1 + IMG_FEATURE_DIM + VECTOR_DIM :,:,:]
        vector = self.conv_vector(vector)
        offset = self.conv_offset(offset)
        feature = torch.cat((img_feature, vector, offset), dim = 1)
        feature = self.conv1(feature)
        feature = nn.ReLU(feature)
        feature = self.conv2(feature)
        feature = nn.ReLU(feature)
        feature = flag * feature
        return feature








