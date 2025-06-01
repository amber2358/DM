import json
import sys
import os
import pickle
import string
import re
import networkx as nx
from collections import defaultdict

project_root = os.path.abspath(os.path.dirname(__file__))
loader_path = os.path.join(project_root, 'loader')
sys.path.append(loader_path)
# 导入数据加载器
from data_loader import PlainDataLoader


def build_graph(class_name):
    """
    构建一个字的有向图，边的权重为字对在语料库中出现的频率
    """
    loader = PlainDataLoader()
    if class_name == "shi":
        corpus = loader.extract_from_multiple(["tangsong","yudingquantangshi","shuimotangshi"])
    else:
        corpus = loader.body_extractor(class_name)

    G = nx.DiGraph()
    edge_weights = defaultdict(int)

    # 用滑动窗口构建图（以字为单位）
    for sentence in corpus:
        for i in range(len(sentence) - 1):
            u = sentence[i]
            v = sentence[i + 1]
            edge_weights[(u, v)] += 1

    # 将边加入图中
    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)

    with open(f"graph_{class_name}.pkl", "wb") as f:
        pickle.dump(G, f)
    
    # 加权 PageRank，考虑出现频率
    pr = nx.pagerank(G, weight='weight')

    with open(f"word_{class_name}.pkl", "wb") as f:
        pickle.dump(pr, f)



def predict_next_char(prefix, class_name, top_k=5):
    """
    根据前缀 prefix 预测下一个字    
    """
    with open(f"./word_data/word_{class_name}.pkl", "rb") as f:
        pr = pickle.load(f)
    with open(f"./word_data/graph_{class_name}.pkl", "rb") as f:
        G = pickle.load(f)    
    
    if not prefix:
        return sorted(pr.items(), key=lambda x: -x[1])[:top_k]
    
    last_char = prefix[-1]
    if last_char not in G:
        return "😢(适合结束这句话了)"

    chinese_punctuation = "，。、！？【】（）《》“”‘’：；——…·"
    all_punctuation = set(string.punctuation + chinese_punctuation)
    
    # 候选：从 last_char 出发的下一个字
    candidates = G[last_char]
    ranked = sorted(
        [(c, pr.get(c, 0)) for c in candidates],
        key=lambda x: -x[1]
    )
    
    rank = ranked[:top_k]  # 取前 top_k 个
    # 只返回字
    related_words =  [char for char, _ in rank if char not in all_punctuation] if rank else []
    return "、".join(sorted(related_words)) if related_words else "😢(适合结束这句话了)"

if __name__ == "__main__":
    class_name = "shi" 
    # build_graph(class_name)
    
    prefix = "春天的花"
    next_chars = predict_next_char(prefix, class_name, top_k=5)
    print(f"前缀 '{prefix}' 的下一个字预测：{next_chars}")