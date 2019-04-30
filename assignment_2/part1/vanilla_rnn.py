################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # std = 1/(num_hidden**0.5)

        self.h_init = torch.zeros(num_hidden, batch_size)

        self.W_hx = nn.Parameter(torch.empty(num_hidden, input_dim))
        nn.init.orthogonal_(self.W_hx)
        #nn.init.uniform_(self.W_hx, -std, std)
        self.W_hh = nn.Parameter(torch.empty(num_hidden, num_hidden))
        nn.init.orthogonal_(self.W_hh)
        #nn.init.uniform_(self.W_hh, -std, std)
        self.b_h = nn.Parameter(torch.zeros(num_hidden, 1))
        self.W_ph = nn.Parameter(torch.empty(num_classes, num_hidden))
        nn.init.orthogonal_(self.W_ph)
        #nn.init.uniform_(self.W_ph, -std, std)
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1))

        self.num_hidden = num_hidden
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device

        # self.rnn = nn.RNN(input_dim, num_hidden, 1, batch_first=True)
        # self.linear = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        # print(x.shape)
        # x = x[:, :, None]
        # rnn_out, self.hidden = self.rnn(x)
        # y_pred = self.linear(rnn_out)
        # return y_pred[:, 4, :]
        h_t = self.h_init
        for i in range(self.seq_length):
            # Reshape into correct form
            x_t = x[:, i]
            x_t = x_t[None, :]
            h_t = torch.tanh(self.W_hx @ x_t + self.W_hh @ h_t + self.b_h)
        p_t = self.W_ph @ h_t + self.b_p
        return torch.t(p_t)
