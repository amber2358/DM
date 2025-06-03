<h1 align="center">
六砚 · 字库
</h1>

<p align="center">
大家好，我们是第八小组，开发了一个用于辅助古诗词创作的智能系统。
</p >

<p align="center">
该系统以中国古诗词数据为基础，融合 PageRank、FP-Growth、Word2Vec、MinHash 及相似度检索等算法，构建了“字-词-句-章”四层级的推荐功能，为用户提供灵感支持与创作引导。
</p >



# DM
## Install
```bash
conda create -n DM python=3.10
conda activate DM
pip install -r requirements.txt
```
## run
```bash
cd src
python app.py
```


## More
```bash
pip install langchain jieba networkx 
```


## Files
- chinese-poetry: 原始数据集
- data
- chroma_data
- word_data:处理后的数据集
- model：权重文件
- src：代码文件
  - app.py: 可视化界面
  - data_loader.py: load类
  - tool_word/frequency.py：数据处理
  - get_similarity: 数据处理
  - task_().py: app.py调用的文件，实现推荐功能
