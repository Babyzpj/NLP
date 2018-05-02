# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: PengjunZhu <1512568691@qq.com>
#
# Function: 使用tf-idf实现获取文档摘要，并使用评估方法ROUGE-2、ROUGE-3进行评估
#      思路：使用IFIDF使用文档中问个句子的权值。然后抽取权值最大的句子作为摘要
# time: 2018.4.07



"""
    具体实现步骤：
       1、对文档进行分句
       2、计算每个句子的TF值、IDF值、TF*IDF值
       3、再跟进IFIDF值对文档排序，抽取前三句作为摘要
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from RougeN import rouge1,rouge2,rouge3
import jieba



def wordTextList(path) :

    """
    :param path:
    :return:
         corpus:[text1,text2...textn], 文本格式：词 词
         corpusList: [[句子，...，句子],[文本,...,句子]]  ,句子格式：词 词 词2
         以上是文本的两种不同表现形式
    """
    corpusList = []
    curpus = []
    summaryList = []


    # 文本
    for item in open(path,'r'):
        itemSummmary = item.split("。")[0]
        summaryList.append(itemSummmary)

        itemText = "。".join(item.split("。")[1:])
        curpus.append(" ".join(jieba.cut(itemText)))

        sentenceList = itemText.strip("").split("。")
        text_sentence = []
        # 句子分词后，加空格转化为字符串
        for itemSentece in sentenceList:
            text_sentence.append(" ".join(jieba.cut(itemSentece)))
        corpusList.append(text_sentence)

    return summaryList, curpus, corpusList


def getAverageValue(sentence):
    """
    :param sentence:是个列表形式，例如sentence = ["I love you"]
    :return: 获取sentece的ifidf均值
    """
    tfidf_test = vectorizer.transform(sentence)
    sentenceTfidfList = tfidf_test.toarray()[0]
    averageValue = sum(sentenceTfidfList) / len(sentenceTfidfList)
    return averageValue

# def SummaryList(pathSummary):
# 	summaryList = []
# 	with open(pathSummary, "r") as file:
# 		for text in file.readlines():
# 			text = text.strip().split("。")
# 			summaryList.append(text[0])
# 		return summaryList


if __name__ == "__main__":

    path = "./PART_II_summary_text.txt"

    summaryList, curpus, corpusList = wordTextList(path)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(curpus)

    # 得到tf-idf的矩阵
    tfidf_train = vectorizer.transform(curpus)

    # 读取摘要列表
    # summaryList = SummaryList(pathSummary)

    Rouge_1_ValueList = []
    Rouge_2_ValueList = []
    Rouge_3_ValueList = []

    for summary, text in zip(summaryList,corpusList):
        valueList = []
        for item in text:
            temp = []
            temp.append(item)
            senteceIfIdfValue = getAverageValue(temp)
            valueList.append(senteceIfIdfValue)
        print(valueList)
        maxIndex = valueList.index(max(valueList))
        # print maxIndex
        summarySentece = "".join(text[maxIndex].strip().split(" "))

        #print("抽取摘要句子：", summarySentece)
        #print("人工摘要句子：", summary)

        S1 = " ".join([summarySentece[j] for j in range(len(summarySentece))])
        print("抽取摘要S1:", S1)


        S2 = " ".join([summary[i] for i in range(len(summary))])
        print("人工摘要：", S2)

        print(rouge1(S1, S2))
        Rouge_1_ValueList.append(rouge1(S1, S2))
        Rouge_2_ValueList.append(rouge2(S1, S2))
        Rouge_3_ValueList.append(rouge3(S1, S2))

    print("Rouge_1_ValueList length:", sum(Rouge_1_ValueList),len(Rouge_1_ValueList))
    print("rouge-1-average =", sum(Rouge_1_ValueList)/len(Rouge_1_ValueList))
    print("rouge-2-average =", sum(Rouge_2_ValueList)/len(Rouge_2_ValueList))
    print("rouge-3-average =", sum(Rouge_3_ValueList)/len(Rouge_3_ValueList))
