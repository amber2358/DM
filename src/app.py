import os
import ast
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from task_frequency import load_data, recommend_keyword
from task_poetry import load_model, recommend_poetry
from task_word import load_graph, recommend_next_char
from task_meaning import build_vectorstore, recommend_sentences

# ---------------------- 数据加载函数 ----------------------
@st.cache_data
def load_all_data(style):
    df = load_data(style.lower())
    p, s = load_model(style.lower())

    embed_model = HuggingFaceEmbeddings(
                # model_name=f'maidalun1020/bce-embedding-base_v1',
                model_name=f'../../bce-embedding-base_v1',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'batch_size': 4, 'normalize_embeddings': False}
            )
    vec = build_vectorstore(load_flag=True, class_name=style.lower(), embed_model=embed_model)
    
    pr, G = load_graph(style.lower())  
    
    return df, p, s, vec, pr, G

# ---------------------- Streamlit 界面 ----------------------
st.set_page_config(page_title="古典体裁词语推荐系统", layout="wide")

# ---------------------- 嵌入网页 ----------------------
with open("./index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

st.markdown("### 🌐 产品展示页面")
components.html("https://the-bird-f.github.io/-/", height=600, scrolling=True)

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
        df, p, s, vec, pr, G  = load_all_data(style)
        
        if method == "推荐下一个字":
            result = recommend_next_char(keyword, pr, G, top_k=5)
            st.success("推荐结果：")
            st.text_area("推荐", result, height=200)
        
        elif method == "推荐主题词语":
            result = recommend_keyword(keyword, df)
            st.success("推荐结果：")
            st.text_area("推荐词语", result, height=200)
            
            
        elif method == "推荐相关诗句":
            result = recommend_sentences(keyword, vec, top_n=5)
            st.success("推荐结果：")
            st.text_area("推荐诗句", result, height=200)
            
        elif method == "推荐相关诗篇":
            result =  recommend_poetry(keyword, p, s, num=3)
            st.success("推荐结果：")
            st.text_area("推荐诗篇", result, height=200)
            
            # st.info("该方法暂未实现，请自行补充函数。")

# ---------------------- 使用说明 ----------------------
with st.expander("使用说明", expanded=False):
    st.markdown("""
    - 输入一个关键词（如“山”、“月”、“风”等），系统会分析该体裁中与之经常共现的词语。
    - 共现关系基于频繁项集挖掘（非语义相似度）。
    - 推荐词语适合用于辅助创作、构思诗意意象、模仿风格。

    **体裁说明：**
    - `chuci`：楚辞（屈原创作风格，想象丰富）
    - `shi`：古诗（以唐诗为主，意境严谨）
    - `songci`：宋词（婉约或豪放，多描情写景）
    - `yuanqu`：元曲（戏剧性强，语言通俗）
    """)
