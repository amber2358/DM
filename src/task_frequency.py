import pandas as pd
import ast
import os

# 加载频繁项集数据
def load_data(style):
    data_dir =  f'./data/{style}'
    
    itemsets_path = os.path.join(data_dir, 'merged_frequent_itemsets.json')
    itemsets_df = pd.read_json(itemsets_path)

    # 如果是字符串格式，转换成列表
    if isinstance(itemsets_df.loc[0, 'itemsets'], str):
        itemsets_df["itemsets"] = itemsets_df["itemsets"].apply(ast.literal_eval)

    return itemsets_df

# 推荐函数：仅返回词语（无支持度、无滑块）
def recommend_keyword(keyword, itemsets_df):

    keyword = keyword.strip()
    related_words = set()

    for _, row in itemsets_df.iterrows():
        items = row["itemsets"]
        if any(keyword in item for item in items):
            for item in items:
                if keyword != item:
                    related_words.add(item)

    return "、".join(sorted(related_words)) if related_words else "😢 该词很可能是新词哦，古人还没用过相关的。"

if __name__ == "__main__":
    style = "songci"
    keyword = "江南"
    df = load_data(style)
    result = recommend_keyword(keyword, df)
    print(f"推荐词语: {result}")