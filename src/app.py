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
def recommend_by_keyword(keyword, style):
    itemsets_df = load_data(style)
    keyword = keyword.strip()
    related_words = set()

    for _, row in itemsets_df.iterrows():
        items = row["itemsets"]
        if any(keyword in item for item in items):
            for item in items:
                if keyword != item:
                    related_words.add(item)

    return "、".join(sorted(related_words)) if related_words else ""

# Gradio 界面
with gr.Blocks(title="古典体裁关联词推荐系统") as demo:
    gr.Markdown("## 🌸 古典体裁词语推荐系统（楚辞 / 诗 / 词）")

    style_selector = gr.Dropdown(choices=["Chuci", "Shi", "Ci"], value="Chuci", label="请选择体裁")
    keyword_input = gr.Textbox(label="请输入关键词", placeholder="如：山、水、芳草")
    output = gr.Textbox(lines=10, label="推荐词语", interactive=False)
    btn = gr.Button("🔍 推荐词语")

    btn.click(fn=recommend_by_keyword, inputs=[keyword_input, style_selector], outputs=output)

demo.launch()
