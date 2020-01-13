"""
non-continuous lstm
all rights reserved by zhang jian
"""
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class ConditionLSTMCell(nn.Module):

    # input_size – The number of expected features in the input x
    # hidden_size – The number of features in the hidden state h
    def __init__(self, input_size, hidden_size, gpu=True):
        super(ConditionLSTMCell, self).__init__()
        self.input_size_raw = input_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        self._gpu = gpu

    # x shape: (batch_size, input_size)
    def forward(self, x, h_0, c_0):
        flag = torch.zeros(x.size(0), 1)
        flag[:, 0] = x.cpu()[:, 0]
        flag = flag.repeat(1, self.hidden_size)
        if self._gpu is True:
            flag = flag.cuda()
        h_1, c_1 = self.lstm_cell(x[:, 1+self.hidden_size : 1+2*self.hidden_size], (h_0, c_0))
        h_2 = torch.add(torch.mul(flag, h_1), torch.mul(h_0, torch.add(torch.neg(flag), 1)))
        c_2 = torch.add(torch.mul(flag, c_1), torch.mul(c_0, torch.add(torch.neg(flag), 1)))
        return h_2, c_2

    # https://discuss.pytorch.org/t/if-else-statement-in-lstm/2550
    # def forward(self, x):
    #     x = self.module1(x)
    #     if (x.data > 0).all():
    #         return self.module2(x)
    #     else:
    #         return self.module3(x)

class ConditionLSTMCell_obin_state(nn.Module):

    # input_size – The number of expected features in the input x
    # hidden_size – The number of features in the hidden state h
    def __init__(self, input_size, hidden_size, gpu=True):
        super(ConditionLSTMCell_obin_state, self).__init__()
        self.hidden_size_raw = hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        self._gpu = gpu

    # x shape: (batch_size, input_size)
    def forward(self, x, h_0, c_0):
        flag = torch.zeros(x.size(0), 1)
        flag[:, 0] = x.cpu()[:, 0]
        flag = flag.repeat(1, self.hidden_size)
        if self._gpu is True:
            flag = flag.cuda()
        h_1, c_1 = self.lstm_cell(x[:, 1:], (h_0, c_0))
        h_2 = torch.add(torch.mul(flag, h_1), torch.mul(h_0, torch.add(torch.neg(flag), 1)))
        c_2 = torch.add(torch.mul(flag, c_1), torch.mul(c_0, torch.add(torch.neg(flag), 1)))
        return h_2, c_2

class ConditionLSTMCell_obin(nn.Module):

    # input_size – The number of expected features in the input x
    # hidden_size – The number of features in the hidden state h
    def __init__(self, input_size, hidden_size, gpu=True):
        super(ConditionLSTMCell_obin, self).__init__()
        self.input_size = input_size                        #input_size 128
        self.hidden_size = hidden_size                      #hidden_size 128
        self.lstm_cell = nn.LSTMCell(self.input_size, hidden_size)
        self._gpu = gpu

    # x shape: (batch_size, input_size)
    def forward(self, x, h_0, c_0):
        flag = torch.zeros(x.size(0), 1)
        flag[:, 0] = x.cpu()[:, 0]
        flag = flag.repeat(1, self.hidden_size)
        if self._gpu is True:
            flag = flag.cuda()
        h_1, c_1 = self.lstm_cell(x[:, 1+self.hidden_size : 1+2*self.hidden_size], (h_0, c_0))
        h_2 = torch.add(torch.mul(flag, h_1), torch.mul(h_0, torch.add(torch.neg(flag), 1)))
        c_2 = torch.add(torch.mul(flag, c_1), torch.mul(c_0, torch.add(torch.neg(flag), 1)))
        return h_2, c_2

    # https://discuss.pytorch.org/t/if-else-statement-in-lstm/2550
    # def forward(self, x):
    #     x = self.module1(x)
    #     if (x.data > 0).all():
    #         return self.module2(x)
    #     else:
    #         return self.module3(x)
    
class ConditionLSTMCell2(nn.Module):
    
    def __init__(self, input_size, hidden_size, gpu=True):
        super(ConditionLSTMCell2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self._gpu = gpu
        
        
    def forward(self, x, h_0, c_0):
        pass



class ConditionLSTMLayer(nn.Module):
    def __init__(self):
        super(ConditionLSTMLayer, self).__init__()

    def forward(self, *input):
        pass


class ConditionLSTM(nn.Module):
    def __init__(self):
        super(ConditionLSTM, self).__init__()

    def forward(self, *input):
        pass


# https://stackoverflow.com/questions/50168224/does-a-clean-and-extendable-lstm-implementation-exists-in-pytorch
class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):

        if hidden is None:
            hidden = self._init_hidden(x)
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)

        h_t = torch.mul(o_t, c_t.tanh())

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)

    @staticmethod
    def _init_hidden(input_):
        h = torch.zeros_like(input_.view(1, input_.size(1), -1))
        c = torch.zeros_like(input_.view(1, input_.size(1), -1))
        return h, c


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)

    def forward(self, input_, hidden=None):
        # input_ is of dimensionalty (1, time, input_size, ...)

        outputs = []
        for x in torch.unbind(input_, dim=1):
            hidden = self.lstm_cell(x, hidden)
            outputs.append(hidden[0].clone())

        return torch.stack(outputs, dim=1)




if __name__ == '__main__':
    # model
    rnn1 = nn.LSTMCell(10, 20)
    rnn2 = ConditionLSTMCell(10, 20)

    optimizer = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    # batch size, time
    batch_size = 3
    time_step = 6
    hidden_size = 20
    input_size = 10
    input = torch.randn(time_step, batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    cx = torch.randn(batch_size, hidden_size)
    output = []

    for i in range(time_step):
        # input[i] of shape (batch, input_size): tensor containing input features
        hx, cx = rnn1(input[i], (hx, cx))
        # output shape: (time_step, batch_size, hidden_size)
        output.append(hx)

    result = output[-1, :, :]
