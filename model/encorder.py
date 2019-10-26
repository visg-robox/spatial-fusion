# Encorder for voxel_feature to  encoded feature
#
# input: (batch_size, near_num, time_step, 1(flag) + img_feature_dim + vector_dim + offset_dim + location_dim)
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

IMG_FEATURE_DIM = common.img_feature_size
ENBEDDING_DIM = IMG_FEATURE_DIM

VECTOR_DIM = common.vector_size
OFFSET_DIM = common.offset_size
LOCATION_DIM = common.location_size

class encorder(nn.Module):
    def __init__(self, qk_dim):
        super(encorder, self).__init__()
        self.conv_img = nn.Conv2d(IMG_FEATURE_DIM, ENBEDDING_DIM, 1)
        self.conv_vector = nn.Conv2d(VECTOR_DIM, ENBEDDING_DIM, 1)
        self.conv_offset = nn.Conv2d(OFFSET_DIM, ENBEDDING_DIM, 1)
        self.conv1 = nn.Conv2d(IMG_FEATURE_DIM + ENBEDDING_DIM * 2, qk_dim, 1)
        self.conv2 = nn.Conv2d(qk_dim, qk_dim, 1)
        self.relu0_1 = nn.LeakyReLU(0.2)
        self.relu0_2 = nn.LeakyReLU(0.2)
        self.relu0_3 = nn.LeakyReLU(0.2)
        self.relu1 = nn.LeakyReLU(0.2)
        self.relu2 = nn.Tanh()

        self.bn1 = nn.BatchNorm2d(ENBEDDING_DIM)
        self.bn2 = nn.BatchNorm2d(ENBEDDING_DIM)
        self.bn3 = nn.BatchNorm2d(ENBEDDING_DIM)
        self.bn4 = nn.BatchNorm2d(ENBEDDING_DIM*2)
        self.bn5 = nn.BatchNorm2d(ENBEDDING_DIM*2)

    def forward(self, input):
        transpose_input = input.permute(0,3,1,2)
        flag = transpose_input[:,:1,:,:]
        img_feature = transpose_input[:,1 : 1 + IMG_FEATURE_DIM,:,:]
        vector = transpose_input[:,1 + IMG_FEATURE_DIM :1 + IMG_FEATURE_DIM + VECTOR_DIM,:,:]
        offset = transpose_input[:,1 + IMG_FEATURE_DIM + VECTOR_DIM:,:,:]
        #location = transpose_input[:,1 + IMG_FEATURE_DIM + VECTOR_DIM + LOCATION_DIM:,:,:]

        img_feature = self.conv_img(img_feature)
        img_feature = self.bn1(img_feature)
        img_feature = self.relu0_1(img_feature)

        vector = self.conv_vector(vector)
        vector = self.bn2(vector)
        vector = self.relu0_2(vector)

        offset = self.conv_offset(offset)
        offset = self.bn3(offset)
        offset = self.relu0_3(offset)

        #location = self.conv_location(location)
        #location = self.relu0_3(location)
        feature_raw = torch.cat((img_feature, vector, offset), dim = 1)
        feature = self.conv1(feature_raw)
        feature = self.bn4(feature)
        feature = self.relu1(feature)
        feature = self.conv2(feature)
        feature = self.bn5(feature)
        feature = self.relu2(feature)
        feature = flag * feature
        return feature,  feature_raw  #[bz, ]








