# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: PengjunZhu <1512568691@qq.com>
#
# Function: 这是代码summary_pagerank.py的另一版，利用w2v和pagerank结合的文章
#
# time:
#

from __future__ import division
import gensim
import jieba
import numpy as np
from numpy import *
from scipy import sparse
import networkx as nx
import scipy
import chardet
import math
import re
import networkx as nx
import operator
from  scipy import *
import pandas as pd

#path = '/home/ZPJ/dataset/summary/Ideal_test_data/dataset_ideal_4.txt'
path = "./Part_III_text_310.txt"

pathSummary = './machineSummary.txt'
manchineSummary = open(pathSummary ,"w+")


model = gensim.models.Word2Vec.load(u'/home/ZPJ/Save_Model/model_dem_50/mode_50dem.model')
# 计算两个向量的余弦值，及相似度
def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

files = open(path, 'r')
p = 0
q = 0
k = 0
m = 0
list_error = []
for file in files:
    # print chardet.detect(file)
    file1 = file.strip().decode("utf-8")
    file2 = file1.strip().strip(u'。')

    # 分词并分以"。"为句子分隔符
    words = jieba.cut(file2)
    file3 = " ".join(words).encode('utf-8')   # 不encode('utf-8')的file就是Unicode码
    
   
    #file3 = "。".join(file3.split("？"))
    #file3 = "。".join(file3.split("！"))
    #file3 = "。".join(file3.split("；")) 
    file_sentence = file3.split('。') # 将文章从句号处分割，得到列表file_sentence，utf-8



    # 删除列表最后一句为空元素(有可能文章最后是分隔符)
    if len(file_sentence[-1]) == 0:
        file_sentence.pop()

    # # 将文章句子少于3句的文章去掉，并统计符合条件文章句子数为p
    length = len(file_sentence)
    #if length < 4:
     #   continue
    p += 1

    sen_no_filter = '。'.join(file_sentence)  # 不带分隔符的句子

    total_sen_vec = []
    text_vec = 50*[0]                    # 定义一个元素为0的50维文档向量
    text_vec_add = []

    for i in range(length):             # 对于k文档的句子进行循环
        file_sentence[i] = file_sentence[i].strip().split(' ') # 去除句子句首句尾的符号

        length1 = len(file_sentence[i])   # 文件的第i个句子的长度
        sen_vec = 50*[0]
        
        for j in range(length1):          # 对于句子的词进行循环
            try:
                sen_array = model[file_sentence[i][j].decode("utf-8")]
                sen_vec = sen_vec +sen_array 
            except:
                pass

        text_vec = text_vec + sen_vec                       # 整个文本各个句子向量叠加后得到的向量
        total_sen_vec.append([i/length1 for i in text_vec])                 #
    # print '文本向量均值：',total_sen_vec

    # print '第%s个文档:' % p, file

    # 构造pagerank的临接矩阵
    adj_max = np.zeros((length,length))
    for m in range(length):
        Ab_value = 0
        for n in range(length):
            if m == n :
                adj_max[m][n] = 0
            elif cos_dist(total_sen_vec[m],total_sen_vec[n])<0:
                adj_max[m][n] = 0
	    elif m+1 == n:
		adj_max[m][n]= 0
            else:
                adj_max[m][n] = cos_dist(total_sen_vec[m],total_sen_vec[n])


          #  elif cos_dist(total_sen_vec[m],total_sen_vec[n]) > 0.5:
           #     adj_max[m][n]=1
                #adj_max[m][n] = cos_dist(total_sen_vec [m],total_sen_vec[n])
   # print '归一化前的临接矩阵为',adj_max


    df = pd.DataFrame(adj_max)
    df = df.apply(lambda i:[item*1.0/sum(i) for item in i],axis=1)
    df = df.fillna(0)  # 用0t
    adj1_max = np.array(df)
    # print '归一化后的临接矩阵为：'
    print adj1_max

    # 将ndarray类型数据，转化为 sparse.csr_matrix类型。然后利用pagerank算法计算句子的重要程度
    sim_matrix = adj1_max.tolist()        # numpy.ndarray类型转化为list类型
    sim_matrix1 = np.matrix(sim_matrix)  # list类型转化为mat矩阵类型
    sim_matrix2 = sparse.csr_matrix(sim_matrix1) # 将矩阵类型转化为nx.from_scipy_sparse_matrix()接受的类型

    nx_graph = nx.from_scipy_sparse_matrix(sim_matrix2)  # 为文本的句子向量构成，每行均为一个句子向量
    scores = nx.pagerank(nx_graph)              # scores是个词典，key:value = 句子索引：句子重要程度
    print scores

    length_n = len(scores)

    #将重要程度排序
    sortedFreq = sorted(scores.items(), key = operator.itemgetter(1), reverse = True)
    top1_sim = int(sortedFreq[0][0])

    print top1_sim 
    print "摘要句为：","".join(file_sentence[top1_sim])
    manchineSummary.write("".join(file_sentence[top1_sim]) + "\n")
  


