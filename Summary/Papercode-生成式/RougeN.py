# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: PengjunZhu <pengjun.zhu@qq.com>
#
# Function:自动文本摘要评测，主要实现Rouge-1、Rouge-2、Rouge-3
#
# time:2018.4.07
#

from __future__ import division
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


"""
参考文档 ：https://blog.csdn.net/qq_25222361/article/details/78694617
"""

# class RougeN:


def rouge1(machineSummary, humanSummary):

    machineSummaryList = machineSummary.split(" ")
    humanSummaryList = set(humanSummary.split(" "))

    length = len(set([item for item in machineSummaryList if item in humanSummaryList]))
    # print("相同的字有：", "".join([item for item in machineSummaryList if item in humanSummaryList]))
    humanLength = len(humanSummaryList)


    rouge1 = round(length/humanLength, 3)
    # print "rouge-1 =", rouge1

    return rouge1


def rouge2(machineSummary, humanSummary):

    machineSummaryList = machineSummary.split(" ")
    humanSummaryList = humanSummary.split(" ")

    BigramMachineList = set([machineSummaryList[i] + " " + machineSummaryList[i+1] for i in range(len(machineSummaryList)-1)])
    BigramhumanList = set([humanSummaryList[j] + " " + humanSummaryList[j+1] for j in range(len(humanSummaryList)-1)])

    bi_length = len(set([bi_item for bi_item in BigramMachineList if bi_item in BigramhumanList]))
    Bi_totalLegth = len(BigramhumanList)

    rouge2 = round(bi_length/Bi_totalLegth,3)
    # print "rouge2 =", rouge2

    return rouge2

def rouge3(machineSummary, humanSummary):

    machineSummaryList = machineSummary.split(" ")
    humanSummaryList = humanSummary.split(" ")

    BigramMachineList = set([machineSummaryList[i] + " " + machineSummaryList[i+1] + " " + machineSummaryList[i+2]
                         for i in range(len(machineSummaryList)-2)])
    BigramhumanList = set([humanSummaryList[j] + " " + humanSummaryList[j+1] + " " + humanSummaryList[j+2]
                       for j in range(len(humanSummaryList)-2)])

    Tri_length = len(set([Tri_item for Tri_item in BigramMachineList if Tri_item in BigramhumanList]))
    Tri_totalLegth = len(BigramhumanList)

    rouge3 = round(Tri_length/Tri_totalLegth,3)
    # print "rouge2 =", rouge3

    return rouge3

# if __name__ == "__main__":
#     # 机器生成的摘要S1, 人工生成的摘要S2
#     S1 = "the cat was found under the bed"
#     S2 = "the cat was under the bed"
#
#     Rouge = RougeN()
#     rouge1 = Rouge.rouge1(S1, S2)
#     rouge2 = Rouge.rouge2(S1, S2)
#     rouge3 = Rouge.rouge3(S1, S2)
#
#     S11 = "猫 被 发 现 在 地 上"
#     S22 = " 猫 在 地 上"
#     rouge1 = Rouge.rouge1(S11, S22)
#     rouge2 = Rouge.rouge2(S11, S22)
#     rouge3 = Rouge.rouge3(S11, S22)