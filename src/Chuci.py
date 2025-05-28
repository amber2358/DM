import sys
import os
import jieba
import re

# 动态添加 loader 所在路径
project_root = os.path.abspath(os.path.dirname(__file__))
loader_path = os.path.join(project_root, 'loader')
sys.path.append(loader_path)

# 导入数据加载器

from data_loader import PlainDataLoader

# 创建加载器
loader = PlainDataLoader()

# 查看所有可用数据集 id_table 映射（可跳过）
# print(loader.id_table)

# 加载楚辞正文内容
# 9: 'chuci'
chuci_data = loader.body_extractor("chuci")
# print(f"共加载 {len(chuci_data)} 行楚辞，内容示意：")
# print(chuci_data[0])

# 定义用于切分的符号，包括：兮，、。；！？等常见标点
def mark_and_clean_line(line):
    # 用这些标点符号统一替换为“|”
    line = re.sub(r'[兮，。、；！？：,.!?;:「」‘’“”()（）《》【】[\]{}<>——\-~·…]', '|', line)
    # 再去除非中文和“|”的内容
    return re.sub(r'[^\u4e00-\u9fa5\|]', '', line)

# 保存分词结果
fenci_data = []
stopwords = set(['兮', '而', '之', '其', '不', '以', '于', '吾', '曰', '也', '与', '为', '余', '者'])

for line_idx, line in enumerate(chuci_data):
    cleaned = mark_and_clean_line(line)
    parts = cleaned.split('|')
    
    # print(f"第{line_idx+1}段原始内容：{line}")
    line_data = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        words = list(jieba.cut(part))
        filtered_words = [w for w in words if w and w not in stopwords]
        line_data.extend(filtered_words)
        # print(f"  分段{i+1}: {words}")
    fenci_data.append(line_data)
    # print("-" * 50)

# 示例查看
print(fenci_data[0])

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import pandas as pd

# 编码分词数据
te = TransactionEncoder()
te_ary = te.fit(fenci_data).transform(fenci_data)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 使用 FP-Growth 进行频繁项挖掘
freq_itemsets = fpgrowth(df, min_support=0.0005, use_colnames=True)

# 排序查看前20个频繁项集
top_freq = freq_itemsets.sort_values(by="support", ascending=False)
print(top_freq)


from mlxtend.frequent_patterns import association_rules

rules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.5)
rules_sorted = rules.sort_values(by=['confidence', 'lift'], ascending=False)
print(rules_sorted.head)

output_path = os.path.join(project_root, '../data/Chuci/merged_frequent_itemsets.json')
freq_itemsets.to_json(output_path, orient='records', force_ascii=False, indent=2)
print(f"已保存频繁项集到：{output_path}")

rules_output_path = os.path.join(project_root, '../data/Chuci/association_rules.csv')
rules_sorted.to_csv(rules_output_path, index=False, encoding='utf-8-sig')