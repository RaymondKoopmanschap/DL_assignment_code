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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.h_init = nn.Parameter(torch.zeros(num_hidden, 1))
        self.W_hx = nn.Parameter(torch.empty(num_hidden, input_dim))
        nn.init.xavier_uniform_(self.W_hx)
        self.W_hh = nn.Parameter(torch.empty(num_hidden, num_hidden))
        nn.init.xavier_uniform_(self.W_hh)
        self.b_h = nn.Parameter(torch.zeros(num_hidden, 1))
        self.W_ph = nn.Parameter(torch.empty(num_classes, num_hidden))
        nn.init.xavier_uniform_(self.W_ph)
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1))

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device

    def forward(self, x):
        h_t = self.h_init
        for i in range(self.seq_length):
            # Reshape into correct form
            x_t = x[:, i]
            x_t = x_t[None, :]
            h_t = torch.tanh(self.W_hx @ x_t + self.W_hh @ h_t + self.b_h)
        p_t = self.W_ph @ h_t + self.b_p
        return torch.t(p_t)
