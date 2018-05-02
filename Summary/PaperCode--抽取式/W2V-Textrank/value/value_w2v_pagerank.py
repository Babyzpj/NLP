# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: PengjunZhu <1512568691@qq.com>
#
# Function: 评估机器提取的摘要和人工摘要在ROUGE-1、ROUGE-2、ROUGE-3的表现
#
# time: 20150418
#

from RougeN import rouge1, rouge2, rouge3

machinePath = "./machineSummary.txt"
huamnPath = "./Part_III_humanSummary_310.txt"

def readSummary(Path):
    SummaryList = []
    with open(Path, 'r') as machineSummaryFile:
        for sentence in machineSummaryFile.readlines():
            SummaryList.append(sentence)
        return SummaryList

if __name__ == "__main__":
    machineSummaryList = readSummary(machinePath)
    humanSummaryList = readSummary(huamnPath)

    Rouge_1_ValueList = []
    Rouge_2_ValueList = []
    Rouge_3_ValueList = []

    
    i = 0
    for machineSummary, humanSummary in zip(machineSummaryList, humanSummaryList):
        i += 0
        if i>1:
            break
        # print "机器摘要："
        # print "人工摘要：", humanSummary, machineSummary

        #for j in range(len(machineSummary)):
        #    print(machineSummary[j])

					
					
        S1 = " ".join([machineSummary[j] for j in range(len(machineSummary))])
        print("抽取摘要S1:", S1)


        S2 = " ".join([humanSummary[i] for i in range(len(humanSummary))])
        print("人工摘要：", S2)



        print(rouge1(S1, S2))
        #
        Rouge_1_ValueList.append(rouge1(S1, S2))
        Rouge_2_ValueList.append(rouge2(S1, S2))
        Rouge_3_ValueList.append(rouge3(S1, S2))
    #
    print("Rouge_1_ValueList length:", sum(Rouge_1_ValueList), len(Rouge_1_ValueList))
    print("rouge-1-average =", sum(Rouge_1_ValueList) / len(Rouge_1_ValueList))
    print("rouge-2-average =", sum(Rouge_2_ValueList) / len(Rouge_2_ValueList))
    print("rouge-3-average =", sum(Rouge_3_ValueList) / len(Rouge_3_ValueList))
