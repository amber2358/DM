<h1 align="center">
六砚 · 字库
</h1>

<p align="center">
大家好，我们是第八小组，开发了一个用于辅助古诗词创作的智能系统。
</p >

<p align="center">
该系统以中国古诗词数据为基础，融合 PageRank、FP-Growth、Word2Vec、MinHash 及相似度检索等算法，构建了“字-词-句-章”四层级的推荐功能，为用户提供灵感支持与创作引导。
</p >

<p align="center">
    <a href="https://github.com/amber2358/DM"> 
        <img alt="Github Code" src="https://img.shields.io/badge/Github-Code-blue"> 
    </a>
    <a href="https://the-bird-f.github.io/-/index.html"> 
        <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-green"> 
    </a>
</p>

## Install
```bash
git clone https://github.com/amber2358/DM.git
cd DM
conda create -n poetry python=3.11
conda activate poetry
pip install -r requirements.txt
```
## run
```bash
streamlit run src/app.py
```

## Dataset
```bash
git clone https://github.com/chinese-poetry/chinese-poetry.git
```


## Files
- chinese-poetry: 原始数据集
- data: 处理后的数据集
- word_data: 处理后的数据集
- chroma_data: 向量数据库
- model：权重文件
- src：代码文件
  - app.py: 可视化界面
  - data_loader.py: load类
  - tool_word/frequency.py：数据处理
  - get_similarity: 数据处理
  - task_*.py: app.py调用的文件，实现推荐功能
