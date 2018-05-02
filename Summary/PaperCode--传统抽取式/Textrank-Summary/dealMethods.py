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

            item = unicode(item, "utf-8")
            Sentence = item.strip().split(u"。")

            SummaryList.append(Sentence[0])
            TextList.append("。".join(Sentence[1:]))

        return SummaryList, TextList
    finally:
        LCSTSCurpus.close()

def readSogouCurpusFile(SogouCurpusPath):
    """
    :param SogouCurpusPath: 
    :return: 文本中所有文本的人工摘要列表，文本所有的正文列表
    """

    SogouCurpus = open(SogouCurpusPath)
    try:
        all_Text = SogouCurpus.readlines()

        summaryList = []
        textList = []

        for i in range(len(all_Text)):
            if i % 3 == 0:

                print chardet.detect(all_Text[i])
                try:
                    summary_item = unicode(all_Text[i], "GB2312")

                except:
                    print all_Text[i]

                summaryList.append(summary_item)
            elif i % 3 == 1:
                print chardet.detect(all_Text[i])
                # text_item = unicode(all_Text[i], "GB2312")

                try:
                    text_item = unicode(all_Text[i], "GB2312")
                except:
                    print all_Text[i]

                textList.append(text_item)
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

def load_stopwords(path='D:\DataSet\PaperDataSet2018\StopWord\StopWords.txt'):
    """ 
    加载停用词 
    :param path: 
    :return: 
    """
    with open(path) as f:
        stopwords = filter(lambda x: x, map(lambda x: x.strip().decode('utf-8'), f.readlines()))
    stopwords.extend([' ', '\t', '\n'])
    return frozenset(stopwords)


def cutWords(text):
    """

    :param text: 
    :return: 对文本分词后的列表，词与词之间为空格
    """
    itemList = []
    segs = jieba.cut(text, cut_all=True)
    for i in segs:
        print i
        itemList.append(i)

    seg = " ".join(itemList)
    print type(seg)
    return seg