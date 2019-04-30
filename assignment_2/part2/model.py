# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        self.lstm = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.embedding = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(self, x, h, c):
        #emb = self.embedding(x)
        lstm_out, (h, c) = self.lstm(x, (h, c))
        # print("size h: " + str(self.hidden.shape))
        # print("size c: " + str(self.cell.shape))
        y_pred = self.linear(lstm_out)
        # print(lstm_out.shape)
        # print(y_pred.shape)
        return y_pred, h, c
        pass
