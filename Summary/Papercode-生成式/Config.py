#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: Config.py
@time: 2018/3/7 14:55
"""

from configparser import ConfigParser

class configer:
    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.read(config_path, encoding='utf-8')

    @property
    def text_train_path(self):
        return self.config.get('path', 'text_train_path')

    @property
    def tgt_train_path(self):
        return self.config.get('path', 'tgt_train_path')

    @property
    def text_test_path(self):
        return self.config.get('path', 'text_test_path')

    @property
    def tgt_test_path(self):
        return self.config.get('path', 'tgt_test_path')

    @property
    def prediction_path(self):
        return self.config.get('path', 'prediction_path')

    @property
    def GRU(self):
        return self.config.getboolean('network', 'GRU')

    @property
    def LSTM(self):
        return self.config.getboolean('network', 'LSTM')

    @property
    def Steps(self):
        return self.config.getint('parameters', 'Steps')

    @property
    def lr(self):
        return self.config.getfloat('parameters', 'lr')

    @property
    def hidden_size(self):
        return self.config.getint('parameters', 'hidden_size')

    @property
    def hidden_layers(self):
        return self.config.getint('parameters', 'hidden_layers')

    @property
    def dropout(self):
        return self.config.getfloat('parameters', 'dropout')

    @property
    def embed_dim(self):
        return self.config.getint('parameters', 'embed_dim')

    @property
    def attn_model(self):
        return self.config.get('parameters', 'attn_model')

    @property
    def max_length(self):
        return self.config.get('parameters', 'max_length')

    @property
    def bidirectional(self):
        return self.config.getboolean('parameters', 'bidirectional')

    @property
    def use_cuda(self):
        return self.config.getboolean('parameters', 'use_cuda')

    @property
    def teacher_forcing(self):
        return self.config.getfloat('parameters', 'teacher_forcing')

    @property
    def clip(self):
        return self.config.getfloat('parameters', 'clip')

    @property
    def SOS_token(self):
        return self.config.getint('parameters', 'SOS_token')

    @property
    def EOS_token(self):
        return self.config.getint('parameters', 'EOS_token')
