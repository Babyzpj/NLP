解释下：
    Part_II.txt是从哈工大提供的文本摘要数据集中提取出来的，提取标准人工评分为5分,共3538个测试文本，每行格式为"人工摘要。文本"
	
	splitTextSummary.py 目的是将Part_II.txt分成两部分，即人工摘要部分Part_II__humanSummary.txt，和文本部分Part_II_text.txt。
	
	
	最后，用我们的摘要系统获取Part_II_text.txt的摘要，然后用value里的评估函数进行评估。