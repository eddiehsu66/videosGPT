# videosGPT
可以得到视频的概述

0 、openai 的 Embedding 接口
问了一下 chatgpt：

1 、文本切割
将文本切割成一小部分，调用 openai 的 embedding 接口，返回这段文本的 embedding 的向量数据。存储这些数据，并且保存好对应关系。

2 、用户提问
将用户提的问题，调用 openai 的 embedding 接口，返回问题的向量数据。

3 、搜索向量
计算相似度。用问题的向量，在之前切割的所有向量数据里，计算和问题向量相似度最高的几个文本(余弦定理)。

4 、调用 chatgpt
准备特殊的 prompt ，里面带上切割的文本内容，加上问题的 prompt 。

例子中的 prompt 是这样的：
