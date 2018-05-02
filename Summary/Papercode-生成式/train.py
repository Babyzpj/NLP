#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: train.py
@time: 2018/3/7 14:55
"""

import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import random
random.seed(23)


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, config):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    input_length = len(input_variable)
    target_length = len(target_variable)

    encoder_hidden = encoder.init_hidden()
    # print(encoder_hidden.size())
    encoder_ouputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[config.SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size*2))
    # decoder_hidden = encoder_hidden
    decoder_hidden = decoder.init_hidden()

    if config.use_cuda:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    using_teacher_forcing = random.random() < config.teacher_forcing
    if using_teacher_forcing:
        for i in range(target_length):
            # print("#", i)
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_ouputs)
            loss += criterion(torch.unsqueeze(decoder_output.view(-1), 0), target_variable[i])
            decoder_input = target_variable[i]
    else:
        for i in range(target_length):
            # print("##", i)
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_ouputs)
            loss += criterion(torch.unsqueeze(decoder_output.view(-1), 0), target_variable[i])
            _, topi = decoder_output.data.topk(1)
            index = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[index]]))
            if config.use_cuda:
                decoder_input = decoder_input.cuda()

            if index == config.EOS_token:
                break
    loss.backward()
    utils.clip_grad_norm(encoder.parameters(), config.clip)
    utils.clip_grad_norm(decoder.parameters(), config.clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss[0] / target_length