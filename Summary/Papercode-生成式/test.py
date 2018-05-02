#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: test.py
@time: 2018/3/8 21:48
"""


import random
import numpy as np
import torch
import jieba

a = torch.randn(1,1,2)
b = torch.randn(1,2,3)
print(a)
print(b)
print(torch.bmm(a, b))


# 。；？！
path = '/Users/zhenranran/Desktop/TrainData.txt'
path_tgt = 'data/train/target.txt'
path_txt = 'data/train/text.txt'
f_tgt = open(path_tgt, 'w', encoding='utf-8')
f_txt = open(path_txt, 'w', encoding='utf-8')

# for line in testtext:
#     line = ' '.join(line)
#     line = jieba.cut(line)
#     line = [item for item in filter(lambda x: x != ' ', line)]
#     line = ' '.join(line)
#     testNewFile.write(line + '\n')


with open(path, 'r', encoding='utf-8') as f:
    text_lists = f.readlines()
    for line in text_lists:
        line = jieba.cut(line)
        line = [item for item in filter(lambda x: x != ' ', line)]
        line = ' '.join(line)
        # print(line)
        for idx in range(len(line)):
            if line[idx] == '。':
                break
            elif line[idx] == '；':
                break
            elif line[idx] == '？':
                break
            elif line[idx] == '！':
                break
        # print(line[:idx+1])
        # print(line[idx+1:])
        print(line[:idx+1], file=f_tgt)
        print(line[idx+1:], file=f_txt, end='')

f_tgt.close()
f_txt.close()
#
#



# path_en = 'D:/Corpus/MT/WMTdata/WMT15/train/train.en'
# path_vi = 'D:/Corpus/MT/WMTdata/WMT15/train/train.vi'
# from urllib.request import urlopen
# response = urlopen(path)
# html = response.read()
# en_num = 0
# en_list = []
# with open(path_en, 'r', encoding='utf-8') as f:
#     en_list = f.readlines()
# vi_list = []


# with open(path_vi, 'r', encoding='utf-8') as f:
#     vi_list = f.readlines()

# flag = True
# while(flag):
#     for i in range(len(en_list)):
#         if i+1 == len(en_list):
#             flag = False
#             break
#         if en_list[i] == '\n':
#             del en_list[i]
#             del vi_list[i]
#             break
# print(len(en_list))
# print(len(vi_list))
# path_en = 'train.en'
# path_vi = 'train.vi'
# with open(path_en, 'w') as f:
#     for line in en_list:
#         print(line, file=f, end='')
# with open(path_vi, 'w', encoding='utf-8') as f:
#     for line in vi_list:
#         print(line, file=f, end='')
#


# a = [1,2,3,4, '', '', 6]
# flag = True
# while(flag):
#     for i in range(len(a)):
#         if a[i] == '':
#             del a[i]
#             break
#         if i+1 == len(a):
#             flag = False
#             break
#
# print(a)
# a = 2
# b = 3
# print(float(a)/b)