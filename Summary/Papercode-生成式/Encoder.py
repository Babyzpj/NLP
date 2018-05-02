#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: Encoder.py
@time: 2018/3/7 14:56
"""

import torch.nn as nn
from torch.autograd import Variable
import torch

class Encoder(nn.Module):
    def __init__(self, word_num, config):
        super(Encoder, self).__init__()
        self.use_cuda = config.use_cuda
        self.input_size = word_num
        self.hidden_size = config.hidden_size
        self.hidden_layers = config.hidden_layers
        self.embed_dim = config.embed_dim
        # self.bidirectional = True if config.bidirectional else False
        self.bidirectional = config.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        # self.hidden_size = self.hidden_size // self.num_directions

        self.embedding = nn.Embedding(self.input_size, self.embed_dim)
        self.GRU = nn.GRU(self.embed_dim, self.hidden_size, self.hidden_layers, bidirectional=self.bidirectional)

    def forward(self, inputs, hidden):
        seq_len = len(inputs)
        # print('inputs: ', inputs)
        inputs = self.embedding(inputs)
        inputs = inputs.view(seq_len, 1, -1)
        inputs, hidden = self.GRU(inputs, hidden)
        # print(inputs.size())
        return inputs, hidden

    def init_hidden(self):
        if self.bidirectional:
            k = 2
        else:
            k = 1
        hidden = Variable(torch.zeros(k * self.hidden_layers, 1, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
        return hidden

