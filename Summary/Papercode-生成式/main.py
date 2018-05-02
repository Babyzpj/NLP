#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: main.py
@time: 2018/3/7 14:56
"""

from Reader import reader
from Config import configer
from Alphabet import Alphabet
import math
import time
from AttnDecoderRNN import AttnDecoderRNN
from train import train
from Encoder import Encoder
import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable
import random
from RougeN import *

MAX_LENGTH = 10
PAD = '<PAD>'
UNK = '<UNK>'

def as_minutes(sec):
    min = math.floor(sec / 60)
    sec -= min * 60
    return '%dm %ds'% (min, sec)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def seq2id(Alphabet, seq_list):
    ids = []
    for line in seq_list:
        new_ids = []
        for word in line:
            index = Alphabet.from_string(word)
            if index != -1:
                new_ids.append(index)
            else:
                index = Alphabet.from_string(PAD)
                new_ids.append(index)
        new_ids.append(config.EOS_token)
        ids.append(new_ids)
    return ids

def filter_sent_label(sent, label):
    return len(sent) < MAX_LENGTH and len(label) < MAX_LENGTH

def filter_sents_labels(sents, labels):
    return [[sents[i], labels[i]] for i in range(len(sents)) if filter_sent_label(sents[i], labels[i])]



if __name__ == '__main__':
    config_path = 'config.cfg'
    config = configer(config_path)
    txt_reader = reader(config.text_train_path, needFresh=True, language='chn')
    tgt_reader = reader(config.tgt_train_path, needFresh=True, language='chn')
    text_sent_list = txt_reader.getData()
    label_sent_list = tgt_reader.getData()
    txt_test_reader = reader(config.text_test_path, needFresh=True, language='chn')
    tgt_test_reader = reader(config.tgt_test_path, needFresh=True, language='chn')
    txt_test_sent_list = txt_test_reader.getData()
    tgr_test_sent_list = tgt_test_reader.getData()
    # print(text_sent_list[:1])
    # print(label_sent_list[:1])
    # sent_label_list = filter_sents_labels(text_sent_list, label_sent_list)



    '''
        create dictionary
    '''
    # print(text_sent_list)
    text_word_state = {'SOS': 10, 'EOS': 10, PAD: 10}
    label_word_state = {'SOS': 10, 'EOS': 10, PAD: 10}
    for line in text_sent_list:
        for word in line:
            if word not in text_word_state:
                text_word_state[word] = 1
            else:
                text_word_state[word] += 1
    for line in label_sent_list:
        for word in line:
            if word not in label_word_state:
                label_word_state[word] = 1
            else:
                label_word_state[word] += 1

    text_word_state[PAD] = 10
    label_word_state[PAD] = 10
    text_word_state[UNK] = 10
    label_word_state[UNK] = 10

    '''
        create Alphabet
    '''
    text_alpha = Alphabet()
    label_alpha = Alphabet()
    text_alpha.initial(text_word_state)
    label_alpha.initial(label_word_state)

    text_alpha.m_b_fixed = True
    label_alpha.m_b_fixed = True

    # print(text_alpha.from_string(PAD))
    # print(text_alpha.id2string)
    print('text word size:', text_alpha.m_size)
    print('label word size:', label_alpha.m_size)
    # print(label_alpha.id2string)

    '''
        seqs to id
    '''
    #train
    text_id_list = seq2id(text_alpha, text_sent_list)
    label_id_list = seq2id(label_alpha, label_sent_list)

    #test
    # text_test_id_list = seq2id(text_alpha, text_sent_list)
    # label_test_id_list = seq2id(label_alpha, label_sent_list)

    encoder = Encoder(text_alpha.m_size, config)
    decoder = AttnDecoderRNN(label_alpha.m_size, config)

    if config.use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # print(encoder)
    # print(decoder)
    lr = config.lr
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    n_epochs = config.Steps
    plot_every = 200
    print_every = 1

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0


    def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
        ids = seq2id(text_alpha, [sentence])
        input_variable = Variable(torch.LongTensor(ids[0]))

        # through encoder
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

        # through decoder
        decoder_input = Variable(torch.LongTensor([[config.SOS_token]]))
        decoder_context = Variable(torch.zeros(1, decoder.hidden_size * 2))
        # decoder_hidden = encoder_hidden
        decoder_hidden = decoder.init_hidden()

        if config.use_cuda:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        decoder_words = []
        for i in range(max_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            _, topi = decoder_output.data.topk(1)
            index = topi[0][0]
            if index == config.EOS_token:
                decoder_words.append('<EOS>')
                break
            else:
                decoder_words.append(label_alpha.id2string[index])

            decoder_input = Variable(torch.LongTensor([[index]]))

            if config.use_cuda:
                decoder_input = decoder_input.cuda()

        return decoder_words

    def evaluate_all(encoder, decoder, text_id_list, target_id_list, flag):
        outputs = []
        # print('random select one sent to show...')
        # index = random.choice(range(len(text_id_list)))
        # # text = text_sent_list[index]
        # # label = label_sent_list[index]
        #
        # text = text_id_list[index]
        # label = target_id_list[index]
        #
        # words_output = evaluate(encoder, decoder, text, len(label))
        # text = ' '.join(text)
        # label = ' '.join(label)
        # output = ' '.join(words_output[:-1])

        # print('>', text)
        # print('=', label)
        # print('<', output)
        # print()


        r1, r2, r3 = 0, 0, 0
        for index in range(len(text_id_list)):
            text = text_id_list[index]
            label = target_id_list[index]
            words_output = evaluate(encoder, decoder, text, len(label))
            label = ' '.join(label)
            output = ' '.join(words_output[:-1])
            r1 += rouge1(output, label)
            r2 += rouge2(output, label)
            r3 += rouge3(output, label)
            outputs.append(output)
        r1 /= len(text_id_list)
        r2 /= len(text_id_list)
        r3 /= len(text_id_list)
        if flag:
            # save test predict sents to file
            print('save test predict sents to file...')
            with open(config.prediction_path, 'w', encoding='utf-8') as f:
                for line in outputs:
                    print(line, file=f)
            print('down')
        return r1, r2, r3
    '''
        start...
    '''
    for epoch in range(n_epochs):
        for index in range(len(text_sent_list)):
        # index = random.choice(range(len(text_sent_list)))
            text = Variable(torch.LongTensor(text_id_list[index]))
            label = Variable(torch.LongTensor(label_id_list[index]))
            if config.use_cuda:
                text = text.cuda()
                label = label.cuda()
            loss = train(text, label, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, config)

            print_loss_total += loss
        print_loss_total /= len(text_sent_list)
        if epoch == 0:
            continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # print('print_loss_avg:', print_loss_avg)
            print_loss_avg = print_loss_avg.cpu()
            # evaluate_all(encoder, decoder, txt_test_sent_list, tgr_test_sent_list)
            flag = False
            if epoch == n_epochs-1:
                flag = True
            r1, r2, r3 = evaluate_all(encoder, decoder, text_sent_list, label_sent_list, flag)
            r1, r2, r3 = r1*100, r2*100, r3*100
            test_r1, test_r2, test_r3 = evaluate_all(encoder, decoder, txt_test_sent_list, tgr_test_sent_list, flag)
            test_r1, test_r2, test_r3 = test_r1*100, test_r2*100, test_r3*100
            print_summary = 'Epoch:%d |time: %s (%d %d%%)|loss: %.4f|train: r1: %.1f%% r2: %.1f%% r3: %.1f%%|test: r1: %.1f%% r2: %.1f%% r3: %.1f%%' \
                                                                        %   (epoch,
                                                                            time_since(start, float(epoch) / n_epochs),
                                                                            epoch, float(epoch) / n_epochs * 100,
                                                                            float(print_loss_avg.data.numpy()),
                                                                            r1, r2, r3, test_r1, test_r2, test_r3)
            print(print_summary)

    fmodel = 'seq2seq-%d.param' % n_epochs
    torch.save([encoder, decoder], fmodel)
    encoder, decoder = torch.load(fmodel)






    # evaluate_all()


