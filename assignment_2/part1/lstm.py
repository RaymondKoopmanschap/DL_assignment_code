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

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.h_init = nn.Parameter(torch.zeros(num_hidden, 1))
        self.c_init = nn.Parameter(torch.zeros(num_hidden, batch_size))

        # g
        self.W_gx = nn.Parameter(torch.randn(num_hidden, input_dim))
        #nn.init.xavier_uniform_(self.W_gx)
        self.W_gh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        #nn.init.xavier_uniform_(self.W_gh)
        self.b_g = nn.Parameter(torch.zeros(num_hidden, 1))

        # i
        self.W_ix = nn.Parameter(torch.randn(num_hidden, input_dim))
        #nn.init.xavier_uniform_(self.W_ix)
        self.W_ih = nn.Parameter(torch.randn(num_hidden, num_hidden))
        #nn.init.xavier_uniform_(self.W_ih)
        self.b_i = nn.Parameter(torch.zeros(num_hidden, 1))

        # f
        self.W_fx = nn.Parameter(torch.randn(num_hidden, input_dim))
        #nn.init.xavier_uniform_(self.W_fx)
        self.W_fh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        #nn.init.xavier_uniform_(self.W_fh)
        self.b_f = nn.Parameter(torch.zeros(num_hidden, 1))

        # o
        self.W_ox = nn.Parameter(torch.randn(num_hidden, input_dim))
        #nn.init.xavier_uniform_(self.W_ox)
        self.W_oh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        #nn.init.xavier_uniform_(self.W_oh)
        self.b_o = nn.Parameter(torch.zeros(num_hidden, 1))

        # p
        self.W_ph = nn.Parameter(torch.randn(num_classes, num_hidden))
        #nn.init.xavier_uniform_(self.W_ph)
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1))

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device

    def forward(self, x):
        h_t = self.h_init
        c_t = self.c_init
        sig = nn.Sigmoid()
        for i in range(self.seq_length):
            # Reshape into correct form
            x_t = x[:, i]
            x_t = x_t[None, :]
            g_t = torch.tanh(self.W_gx @ x_t + self.W_gh @ h_t + self.b_g)
            i_t = sig(self.W_ix @ x_t + self.W_ih @ h_t + self.b_i)
            f_t = sig(self.W_fx @ x_t + self.W_fh @ h_t + self.b_f)
            o_t = sig(self.W_ox @ x_t + self.W_oh @ h_t + self.b_o)

            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t
        p_t = self.W_ph @ h_t + self.b_p
        return torch.t(p_t)