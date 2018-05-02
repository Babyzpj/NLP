# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: PengjunZhu <1512568691@qq.com>
#
# Function: 使用Textrank实现获取文档摘要，并使用评估方法ROUGE-2、ROUGE-3进行评估
#
# time: 2018.4.07
#

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import jieba
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

import chardet
from dealMethods import cut_sentence, load_stopwords, readLcstsFile,readSogouCurpusFile
from RougeN import rouge1,rouge2,rouge3

def get_abstract(content, size=3):
    """ 
    利用textrank提取摘要 
    :param content: 
    :param size: 
    :return: 
    """
    docs = list(cut_sentence(content))
    tfidf_model = TfidfVectorizer(tokenizer=jieba.cut, stop_words=load_stopwords())
    tfidf_matrix = tfidf_model.fit_transform(docs)
    normalized_matrix = TfidfTransformer().fit_transform(tfidf_matrix)
    similarity = nx.from_scipy_sparse_matrix(normalized_matrix * normalized_matrix.T)
    scores = nx.pagerank(similarity)
    tops = sorted(scores.iteritems(), key=lambda x: x[1], reverse=True)
    size = min(size, len(docs))
    indices = map(lambda x: x[0], tops)[:size]
    return map(lambda idx: docs[idx], indices)

if __name__ == "__main__":

    LcstsPath = "D:\DataSet\PaperDataSet2018\LCSTS_curpus_2.txt"
    # SogouCurpusPath = "D:\DataSet\PaperDataSet2018\Sougou_curpus.txt"

    LcstsSummaryList, LcstsTextList = readLcstsFile(LcstsPath)
    # SogouSummaryList, SogouTextList = readSogouCurpusFile(SogouCurpusPath)


    Rouge_1_ValueList = []
    Rouge_2_ValueList = []
    Rouge_3_ValueList = []

    for Isummary, text in zip(LcstsSummaryList, LcstsTextList):
    # for Isummary, text in zip(SogouSummaryList, SogouTextList):

        S2 = " ".join([Isummary[i] for i in range(len(Isummary))])
        print "人工摘要：", S2

        # 获取数据集中每篇文章的摘要，并append列表里
        humansummary = [sentence for sentence in get_abstract(text)]

        # 抽取最重要一句为摘要，或抽取前3句为摘要
        S1 = humansummary[0]
        # S1 = "".join(humansummary)

        S1 = " ".join([S1[j] for j in range(len(S1))])
        print "抽取摘要S1:", S1

        Rouge_1_ValueList.append(rouge1(S1, S2))
        Rouge_2_ValueList.append(rouge2(S1, S2))
        Rouge_3_ValueList.append(rouge3(S1, S2))

    print "Rouge_1_ValueList length:", sum(Rouge_1_ValueList),len(Rouge_1_ValueList)
    print "rouge-1-average =", sum(Rouge_1_ValueList)/len(Rouge_1_ValueList)
    print "rouge-2-average =", sum(Rouge_2_ValueList)/len(Rouge_2_ValueList)
    print "rouge-3-average =", sum(Rouge_3_ValueList)/len(Rouge_3_ValueList)

