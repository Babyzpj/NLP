# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: PengjunZhu <1512568691@qq.com>
#
# Function:
#
# time:
#
from sklearn.feature_extraction.text import TfidfVectorizer
#
# X_train = ['This is the first document hah.', 'This is the second document.']
# X_test = ['This is the third document.']
#
# vectorizer = TfidfVectorizer()
#
# # 用X_train数据来fit
# vectorizer.fit(X_train)
#
# # 得到tfidf的矩阵
# tfidf_train = vectorizer.transform(X_train)
#
# tfidf_test = vectorizer.transform(X_test)
#
# sentenceTfidfList = tfidf_test.toarray()[0]
# print "测试：",sentenceTfidfList
#
# averageValue = sum(sentenceTfidfList)/len(sentenceTfidfList)

from RougeN import rouge1,rouge2,rouge3
import jieba
import chardet
pathLcsts = './LCSTS_PartII_Text_2.txt'
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

    # 文本
    for itemText in open(path,'r'):
        curpus.append(" ".join(jieba.cut(itemText)))

        sentenceList = itemText.strip("").split("。")
        text_sentence = []
        # 句子分词后，加空格转化为字符串
        for itemSentece in sentenceList:
            text_sentence.append(" ".join(jieba.cut(itemSentece)))
        corpusList.append(text_sentence)

    # print "列表总长1：", len(curpus)
    # print "列表总长2：", len(corpusList)
    return curpus, corpusList


def getAverageValue(sentence):
    """
    :param sentence:是个列表形式，例如sentence = ["I love you"]
    :return: 获取sentece的ifidf均值
    """
    tfidf_test = vectorizer.transform(sentence)
    sentenceTfidfList = tfidf_test.toarray()[0]
    averageValue = sum(sentenceTfidfList) / len(sentenceTfidfList)
    return averageValue

def SummaryList():
	path = './LCSTS_PartII_summary_2.txt'
	summaryList = []
	with open(path, "r") as file:
		for text in file.readlines():
			text = text.strip().split("。")
			summaryList.append(text[0])
		return summaryList


if __name__ == "__main__":
    curpus, corpusList = wordTextList(pathLcsts)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(curpus)

    # 得到tf-idf的矩阵
    tfidf_train = vectorizer.transform(curpus)

    # 读取摘要列表
    summaryList = SummaryList()

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
