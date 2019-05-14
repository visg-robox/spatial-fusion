# Encorder for voxel_feature to  encoded feature
#
# input: (batch_size, near_num, time_step, 1(flag) + img_feature_dim + vector_dim + offset_dim + location_dim)
# output:(batch_size, qk_dim, near_num, time_step)
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

IMG_FEATURE_DIM = common.img_feature_size
ENBEDDING_DIM = IMG_FEATURE_DIM

VECTOR_DIM = common.vector_size
OFFSET_DIM = common.offset_size
LOCATION_DIM = common.location_size


class encorder(nn.Module):
    def __init__(self, qk_dim):
        super(encorder, self).__init__()
        self.conv_vector = nn.Conv2d(VECTOR_DIM, ENBEDDING_DIM, 1)
        self.conv_offset = nn.Conv2d(OFFSET_DIM, ENBEDDING_DIM, 1)
        self.conv_location = nn.Conv2d(LOCATION_DIM, ENBEDDING_DIM, 1)
        self.conv1 = nn.Conv2d(IMG_FEATURE_DIM * 4, qk_dim, 1)
        self.conv2 = nn.Conv2d(qk_dim, qk_dim, 1)
        self.relu1 = nn.Tanh()
        self.relu2 = nn.Tanh()



    def forward(self, input):
        transpose_input = input.permute(0,3,1,2)
        flag = transpose_input[:,:1,:,:]
        img_feature = transpose_input[:,1 : 1 + IMG_FEATURE_DIM,:,:]
        vector = transpose_input[:,1 + IMG_FEATURE_DIM :1 + IMG_FEATURE_DIM + VECTOR_DIM,:,:]
        offset = transpose_input[:,1 + IMG_FEATURE_DIM + VECTOR_DIM :1 + IMG_FEATURE_DIM + VECTOR_DIM + LOCATION_DIM,:,:]
        location = transpose_input[:,1 + IMG_FEATURE_DIM + VECTOR_DIM + LOCATION_DIM:,:,:];

        vector = self.conv_vector(vector)
        vector = self.relu1(vector)

        offset = self.conv_offset(offset)
        offset = self.relu2(offset)

        location = self.conv_location(location)
        location = self.relu2(location)
        feature = torch.cat((img_feature, vector, offset, location), dim = 1)
        feature = self.conv1(feature)
        feature = self.relu1(feature)
        feature = self.conv2(feature)
        feature = self.relu2(feature)
        feature = flag * feature
        return feature   #[bz, ]








