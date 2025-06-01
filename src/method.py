# 封装函数
import gradio as gr
import pandas as pd
import ast
import os

# 加载频繁项集数据
def load_data(style):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, f'../data/{style}')
    
    itemsets_path = os.path.join(data_dir, 'merged_frequent_itemsets.json')
    itemsets_df = pd.read_json(itemsets_path)

    # 如果是字符串格式，转换成列表
    if isinstance(itemsets_df.loc[0, 'itemsets'], str):
        itemsets_df["itemsets"] = itemsets_df["itemsets"].apply(ast.literal_eval)
    return itemsets_df

# 推荐函数：仅返回词语（无支持度、无滑块）
def recommend_by_freq(keyword, style):
    itemsets_df = load_data(style)
    keyword = keyword.strip()
    related_words = set()

    for _, row in itemsets_df.iterrows():
        items = row["itemsets"]
        if any(keyword in item for item in items):
            for item in items:
                if keyword != item:
                    related_words.add(item)

    return "、".join(sorted(related_words)) if related_words else "😢 未找到关联词语，请尝试其他关键词。"

# similar
from data_loader import PlainDataLoader
from shingling import Shingles
from typing import Dict, List, Union
import pickle
import argparse
from get_similarity import Get_Similarity

def recommend_by_sim(dataset, text: Union[str,List[str]], num:int = 3) -> List[List[str]]:
    recommended = []
    loader = PlainDataLoader()
    poems = loader.extract_full_poem(dataset)
    shingler = Shingles()
    text = text.split()
    text = shingler.shingling_sentence(text)
    if dataset not in ['songci', 'yuanqu', 'yudingquantangshi', 'chuci']:
        raise TypeError()
    with open(f"./models/{dataset}_model.pkl", "rb") as file:
        similarer = pickle.load(file)
    for id in similarer.find_similar_poem(text,num):
        recommended.append(''.join(poems[id]))

    return '\n'.join(recommended)