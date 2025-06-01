import streamlit as st
import pandas as pd
import ast
import os
import streamlit.components.v1 as components

# ---------------------- 数据加载函数 ----------------------
@st.cache_data
def load_data(style):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, f'../data/{style}')
    itemsets_path = os.path.join(data_dir, 'merged_frequent_itemsets.json')
    itemsets_df = pd.read_json(itemsets_path)

    if isinstance(itemsets_df.loc[0, 'itemsets'], str):
        itemsets_df["itemsets"] = itemsets_df["itemsets"].apply(ast.literal_eval)

    return itemsets_df

# ---------------------- 推荐函数示例 ----------------------
def recommend_by_keyword(keyword, itemsets_df):
    keyword = keyword.strip()
    related_words = set()

    for _, row in itemsets_df.iterrows():
        items = row["itemsets"]
        if any(keyword in item for item in items):
            for item in items:
                if keyword != item:
                    related_words.add(item)

    return "、".join(sorted(related_words)) if related_words else "未找到关联词语，请尝试其他关键词。"

# ---------------------- Streamlit 界面 ----------------------
st.set_page_config(page_title="古典体裁词语推荐系统", layout="wide")
# ---------------------- 嵌入网页 ----------------------

with open("./index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

st.markdown("### 🌐 产品展示页面")
components.html(html_content, height=600, scrolling=True)

# CSS样式放大字体和调整左栏header
st.markdown("""
    <style>
    .left-header {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .left-label, label {
        font-size: 20px !important;
    }
    .stTextInput>div>div>input {
        font-size: 18px !important;
        height: 38px !important;
    }
    .stButton>button {
        font-size: 18px !important;
        height: 40px !important;
        padding: 6px 20px !important;
    }
    .stTextArea>div>textarea {
        font-size: 18px !important;
    }
    h2 {
        font-size: 36px !important;
    }
    p {
        font-size: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h2 style='text-align: center;'>古典体裁词语推荐系统</h2>
    <p style='text-align: center; color: gray;'>支持《楚辞》/ 古诗 / 宋词，推荐常见共现词</p>
    <hr>
""", unsafe_allow_html=True)

# ----------- 左右分栏 --------------
left_col, right_col = st.columns([0.7, 2.3])  # 左栏窄，右栏宽

with left_col:
    st.markdown("<div class='left-header'>设置</div>", unsafe_allow_html=True)
    style = st.selectbox("请选择体裁", ["Chuci", "Shi", "Ci"], index=0)
    method = st.radio(
        "选择推荐方法",
        ["关联词推荐", "支持度排序", "关键词扩展", "自定义方法"],
        index=0
    )

with right_col:
    keyword = st.text_input("请输入关键词", value="", placeholder="如：山、水、芳草")
    run = st.button("开始推荐")

    if run:
        df = load_data(style)
        if method == "关联词推荐":
            result = recommend_by_keyword(keyword, df)
            st.success("推荐结果：")
            st.text_area("推荐词语", result, height=200)
        else:
            st.info("该方法暂未实现，请自行补充函数。")

# ---------------------- 使用说明 ----------------------
with st.expander("使用说明", expanded=False):
    st.markdown("""
    - 输入一个关键词（如“山”、“月”、“风”等），系统会分析该体裁中与之经常共现的词语。
    - 共现关系基于频繁项集挖掘（非语义相似度）。
    - 推荐词语适合用于辅助创作、构思诗意意象、模仿风格。

    **体裁说明：**
    - `Chuci`：楚辞（屈原创作风格，想象丰富）
    - `Shi`：古诗（以唐诗为主，意境严谨）
    - `Ci`：宋词（婉约或豪放，多描情写景）
    """)
