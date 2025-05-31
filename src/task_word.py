import json
import sys
import os
import pickle
import string
import re
import networkx as nx


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