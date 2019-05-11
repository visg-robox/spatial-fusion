"""
non-continuous lstm
all rights reserved by zhang jian
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.lstm_cell import *


class SSNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, gpu=True):
        super(SSNet, self).__init__()
        # parameter
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._gpu = gpu
        # model
        self.lstm_cell = ConditionLSTMCell(input_size, hidden_size)
        self.lstm_cell2 = ConditionLSTMCell(hidden_size, hidden_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(output_size, affine=False)
        self.init_linear()

    def forward(self, input_data, time_step):
        # input_data shape: (batch_size, time_step, input_size)
        h0_t = Variable(torch.zeros(input_data.size(0), self._hidden_size), requires_grad=True)
        c0_t = Variable(torch.zeros(input_data.size(0), self._hidden_size), requires_grad=True)
        h1_t = Variable(torch.zeros(input_data.size(0), self._hidden_size), requires_grad=True)
        c1_t = Variable(torch.zeros(input_data.size(0), self._hidden_size), requires_grad=True)
        flag = torch.zeros(input_data.size(0), 1)
        if self._gpu is True:
            h0_t = h0_t.cuda()
            c0_t = c0_t.cuda()
            h1_t = h1_t.cuda()
            c1_t = c1_t.cuda()
        for i in range(time_step):
            h0_t, c0_t = self.lstm_cell(input_data[:, i, :], h0_t, c0_t)
            flag[:, 0] = input_data[:, i, 0]
            input_data2 = torch.cat((flag, h0_t.cpu()), 1)
            if self._gpu is True:
                input_data2 = input_data2.cuda()
            h1_t, c1_t = self.lstm_cell2(input_data2, h1_t, c1_t)
        outputs = self.linear(h1_t)
        return outputs

    def freeze_linear(self):
        for p in self.linear.parameters():
            p.requires_grad = False

    def free_linear(self):
        for p in self.linear.parameters():
            p.requires_grad = True

    def init_linear(self):
        weights, bias = load_icnet_parameter()
        self.freeze_linear()
        self.linear.weight.data = torch.from_numpy(weights[0, 0, :, :].T)
        self.linear.bias.data = torch.from_numpy(bias)


class SSNetCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, gpu=True):
        super(SSNetCell, self).__init__()
        # parameter
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._gpu = gpu
        # model
        self.lstm_cell = ConditionLSTMCell(input_size, hidden_size)
        self.lstm_cell2 = ConditionLSTMCell(hidden_size, hidden_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(output_size, affine=False)
        # self.init_linear()

    def forward(self, input_data, h0_t=None, c0_t=None, h1_t=None, c1_t=None):
        # input_data shape: (batch_size, time_step, input_size)
        if h0_t is None:
            h0_t = Variable(torch.zeros(input_data.size(0), self._hidden_size), requires_grad=True)
        if c0_t is None:
            c0_t = Variable(torch.zeros(input_data.size(0), self._hidden_size), requires_grad=True)
        if h1_t is None:
            h1_t = Variable(torch.zeros(input_data.size(0), self._hidden_size), requires_grad=True)
        if c1_t is None:
            c1_t = Variable(torch.zeros(input_data.size(0), self._hidden_size), requires_grad=True)
        flag = torch.zeros(input_data.size(0), 1)
        if self._gpu is True:
            h0_t = h0_t.cuda()
            c0_t = h0_t.cuda()
            h1_t = h1_t.cuda()
            c1_t = c1_t.cuda()
        time_step = input_data.size(1)
        output_lsit = []
        for step in range(time_step):
            h0_t, c0_t = self.lstm_cell(input_data[:, step, :], h0_t, c0_t)
            flag[:, 0] = input_data[:, step, 0]
            input_data2 = torch.cat((flag, h0_t.cpu()), 1)
            if self._gpu is True:
                input_data2 = input_data2.cuda()
            h1_t, c1_t = self.lstm_cell2(input_data2, h1_t, c1_t)
            output = self.linear(h1_t)
            output_lsit.append(output)
        last_output = output_lsit[-1]
        return last_output, h0_t, c0_t, h1_t, c1_t
    
# basic
class Rnn(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS):
        super(Rnn, self).__init__()
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers
        self.rnn = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,  # rnn hidden unit
                num_layers=NUM_LAYERS,
                batch_first=True,  # 设置数据的
         )
        self.out = nn.Linear(128, 13)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


def load_icnet_parameter():
    file_path = '/home/zhangjian/code/project/spatial-fusion/icnet_para'
    biasis_file = file_path + '/cls_biasis.npy'
    weight_file = file_path + '/cls_weights.npy'
    biasis = np.load(biasis_file)
    weights = np.load(weight_file)
    return weights, biasis

# cal weight for Q between K


if __name__ == "__main__":
    pass

