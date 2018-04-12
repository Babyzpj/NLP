# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: PengjunZhu <1512568691@qq.com>
#
# Function: 实现自动摘要处理文本的方法：分句、分词、去停用词、
# time: 2018.04.17
#

import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import chardet
import re

# LcstsPath= "D:\DataSet\PaperDataSet2018\LCSTS_curpus.txt"
def readLcstsFile(LcstsPath):

    """
    :param LcstsPath: 
    :return: 文本中所有文本的人工摘要列表，文本所有的正文列表
    """

    LCSTSCurpus = open(LcstsPath, 'r')

    try:
        SummaryList = []
        TextList = []

        all_Text = LCSTSCurpus.readlines()

        for item in all_Text:

            Sentence = item.strip().split("。")

            # print "摘要：", Sentence[0]
            # print "len(Sentence[1:] =",len(Sentence[1:])

            SummaryList.append(Sentence[0])
            TextList.append("。".join(Sentence[1:]))

        return SummaryList, TextList
    finally:
        LCSTSCurpus.close()


# SogouCurpusPath = "D:\DataSet\PaperDataSet2018\Sougou_curpus.txt"
def readSogouCurpusFile(SogouCurpusPath):
    """
    :param SogouCurpusPath: 
    :return: 文本中所有文本的人工摘要列表，文本所有的正文列表
    """

    SogouCurpus = open(SogouCurpusPath)
    try:
        all_Text = SogouCurpus.read()

        summaryList = []
        textList = []

        for i in range(len(all_Text)):
            if i % 3 == 0:
                summaryList.append(all_Text[i])
            elif i % 3 == 1:
                textList.append(all_Text)
            else:
                pass

        return summaryList, textList
    finally:
        SogouCurpus.close()



def cut_sentence(sentence):
    """ 
    分句 
    :param sentence: 
    :return: 
    """
    if not isinstance(sentence, unicode):
        sentence = sentence.decode('utf-8')
    delimiters = frozenset(u'。！？；')
    buf = []
    for ch in sentence:
        buf.append(ch)
        if delimiters.__contains__(ch):
            yield ''.join(buf)
            buf = []
    if buf:
        yield ''.join(buf)

def cut_words(sentence):
    """ 
    分词 
    :param sentence: 
    :return: 
    """
    stopwords = load_stopwords()
    return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))

def load_stopwords(path='D:\pycharmworkspace\PaperCode2018\Datasets\StopWords.txt'):
    """ 
    加载停用词 
    :param path: 
    :return: 
    """
    with open(path) as f:
        stopwords = filter(lambda x: x, map(lambda x: x.strip().decode('utf-8'), f.readlines()))
    stopwords.extend([' ', '\t', '\n'])
    return frozenset(stopwords)