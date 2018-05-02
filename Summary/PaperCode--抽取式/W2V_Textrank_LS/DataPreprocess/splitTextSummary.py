# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: PengjunZhu <1512568691@qq.com>
#
# Function: 从文本中获取的人工摘要和自动摘要
#
# time:
#

humanSuammaryPath = "./Part_II__humanSummary.txt"
humanSuammary = open(humanSuammaryPath, "w")

textPath = "./Part_II_text.txt"
text = open(textPath, "w")


if __name__ == "__main__":

    path = "./PART_II.txt"
    with open(path, "r") as textFile:
        for item in textFile.readlines():

            itemSummary = item.strip().split("。")[0]
            itemText = "。".join(item.strip().split("。")[1:])

            humanSuammary.write(itemSummary + "\n")
            text.write(itemText + "\n")


