import sys
import os
import jieba
import re
import pandas as pd
from collections import defaultdict
from functools import reduce
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules


from data_loader import PlainDataLoader



if __name__ == "__main__":
    class_name = "chuci"
    loader = PlainDataLoader()
    project_root = ""
    if class_name == "chuci":   
        # 9: 'chuci'
        chuci_data = loader.body_extractor("chuci")
        # print(f"共加载 {len(chuci_data)} 行楚辞，内容示意：")
        # print(chuci_data[0])

        def mark_and_clean_line(line):
            line = re.sub(r'[兮，。、；！？：,.!?;:「」‘’“”()（）《》【】[\]{}<>——\-~·…]', '|', line)
            return re.sub(r'[^\u4e00-\u9fa5\|]', '', line)

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

        print(fenci_data[0])

        te = TransactionEncoder()
        te_ary = te.fit(fenci_data).transform(fenci_data)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        freq_itemsets = fpgrowth(df, min_support=0.0005, use_colnames=True)

        top_freq = freq_itemsets.sort_values(by="support", ascending=False)
        print(top_freq)

        rules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.5)
        rules_sorted = rules.sort_values(by=['confidence', 'lift'], ascending=False)
        print(rules_sorted.head)

        output_path = os.path.join(project_root, '../data/Chuci/merged_frequent_itemsets.json')
        freq_itemsets.to_json(output_path, orient='records', force_ascii=False, indent=2)
        print(f"已保存频繁项集到：{output_path}")

        rules_output_path = os.path.join(project_root, '../data/Chuci/association_rules.csv')
        rules_sorted.to_csv(rules_output_path, index=False, encoding='utf-8-sig')
    
    elif class_name == "songci":
        # 5: 'songci'
        Ci_data = loader.body_extractor("songci")
        print(f"共加载 {len(Ci_data)} 行词，内容示意：")
        print(Ci_data[0])

        def mark_and_clean_line(line):
            line = re.sub(r'[，。、；！？：,.!?;:「」‘’“”()（）《》【】[\]{}<>——\-~·…]', '|', line)
            return re.sub(r'[^\u4e00-\u9fa5\|]', '', line)

        fenci_data = []
        path1 = os.path.join(project_root, '../data/Ci/fenci_shi.txt')
        if not os.path.exists(path1):
            for line_idx, line in enumerate(Ci_data):
                cleaned = mark_and_clean_line(line)
                parts = cleaned.split('|')
                
                # print(f"第{line_idx+1}段原始内容：{line}")
                line_data = []
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue
                    words = list(jieba.cut(part))
                    line_data.extend(words)
                    # print(f"  分段{i+1}: {words}")
                fenci_data.append(line_data)
                # print("-" * 50)
            print(fenci_data[0])

            with open(path1, 'w', encoding='utf-8') as f:
                for line in fenci_data:
                    f.write(' '.join(line) + '\n')
        else:
            with open(path1, 'r', encoding='utf-8') as f:
                fenci_data = [line.strip().split() for line in f if line.strip()]


        def run_fpgrowth_in_batches(fenci_data, batch_size=30000, min_support=0.001):
            all_itemsets = []

            num_batches = (len(fenci_data) + batch_size - 1) // batch_size
            for i in range(num_batches):
                batch = fenci_data[i * batch_size : (i + 1) * batch_size]
                print(f"Processing batch {i+1}/{num_batches} with {len(batch)} transactions")

                te = TransactionEncoder()
                te_ary = te.fit(batch).transform(batch)
                df_batch = pd.DataFrame(te_ary, columns=te.columns_)

                freq_batch = fpgrowth(df_batch, min_support=min_support, use_colnames=True)
                freq_batch['batch_size'] = len(batch)
                all_itemsets.append(freq_batch)

            return all_itemsets


        batch_results = run_fpgrowth_in_batches(fenci_data, batch_size=10000, min_support=0.001)

        itemset_counter = defaultdict(float)
        total_transactions = 0

        for df in batch_results:
            batch_size = df['batch_size'].iloc[0]
            total_transactions += batch_size
            for _, row in df.iterrows():
                key = frozenset(row['itemsets'])
                itemset_counter[key] += row['support'] * batch_size

        merged_freq_itemsets = pd.DataFrame([
            {'itemsets': set(k), 'support': v / total_transactions}
            for k, v in itemset_counter.items()
        ])

        merged_freq_itemsets = merged_freq_itemsets.sort_values(by='support', ascending=False)
        print(merged_freq_itemsets.head(20))

        try:
            rules = association_rules(merged_freq_itemsets, metric="confidence", min_threshold=0.5)
            rules_sorted = rules.sort_values(by=['confidence', 'lift'], ascending=False)
            print(rules_sorted.head(10))
        except Exception as e:
            print("生成关联规则失败，原因：", e)

        output_path = os.path.join(project_root, '../data/Ci/merged_frequent_itemsets.json')
        merged_freq_itemsets.to_json(output_path, orient='records', force_ascii=False, indent=2)
        print(f"已保存频繁项集到：{output_path}")

        rules_output_path = os.path.join(project_root, '../data/Ci/association_rules.csv')
        rules_sorted.to_csv(rules_output_path, index=False, encoding='utf-8-sig')
        
    elif class_name == "shi":
        Shi_data = loader.extract_from_multiple(["tangsong","yudingquantangshi","shuimotangshi"])
        print(f"共加载 {len(Shi_data)} 行诗，内容示意：")
        print(Shi_data[0])

        path1 = os.path.join(project_root, '../data/Shi/simplified_shi.txt')
        if not os.path.exists(path1):
            from opencc import OpenCC
            cc = OpenCC('t2s')  
            Simplified_data = [cc.convert(line) for line in Shi_data]
            print(Simplified_data[0])

            with open(path1, 'w', encoding='utf-8') as f:
                for line in Simplified_data:
                    f.write(line.strip() + '\n')

            print(f"✅ 繁体转简体后的文本已保存至: {path1}")
        else:
            with open(path1, 'r', encoding='utf-8') as f:
                Simplified_data = [line.strip() for line in f if line.strip()]


        def mark_and_clean_line(line):
            line = re.sub(r'[，。、；！？：,.!?;:「」‘’“”()（）《》【】[\]{}<>——\-~·…]', '|', line)
            return re.sub(r'[^\u4e00-\u9fa5\|]', '', line)

        fenci_data = []
        path2 = os.path.join(project_root, '../data/Shi/fenci_shi.txt')
        if not os.path.exists(path2):
            for line_idx, line in enumerate(Simplified_data):
                cleaned = mark_and_clean_line(line)
                parts = cleaned.split('|')
                
                # print(f"第{line_idx+1}段原始内容：{line}")
                line_data = []
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue
                    words = list(jieba.cut(part))
                    line_data.extend(words)
                    # print(f"  分段{i+1}: {words}")
                fenci_data.append(line_data)
                # print("-" * 50)
            print(fenci_data[0])

            with open(path2, 'w', encoding='utf-8') as f:
                for line in fenci_data:
                    f.write(' '.join(line) + '\n')
        else:
            with open(path2, 'r', encoding='utf-8') as f:
                fenci_data = [line.strip().split() for line in f if line.strip()]


        def run_fpgrowth_in_batches(fenci_data, batch_size=30000, min_support=0.001):
            all_itemsets = []

            num_batches = (len(fenci_data) + batch_size - 1) // batch_size
            for i in range(num_batches):
                batch = fenci_data[i * batch_size : (i + 1) * batch_size]
                print(f"Processing batch {i+1}/{num_batches} with {len(batch)} transactions")

                te = TransactionEncoder()
                te_ary = te.fit(batch).transform(batch)
                df_batch = pd.DataFrame(te_ary, columns=te.columns_)

                freq_batch = fpgrowth(df_batch, min_support=min_support, use_colnames=True)
                freq_batch['batch_size'] = len(batch)
                all_itemsets.append(freq_batch)

            return all_itemsets

        batch_results = run_fpgrowth_in_batches(fenci_data, batch_size=10000, min_support=0.001)

        itemset_counter = defaultdict(float)
        total_transactions = 0

        for df in batch_results:
            batch_size = df['batch_size'].iloc[0]
            total_transactions += batch_size
            for _, row in df.iterrows():
                key = frozenset(row['itemsets'])
                itemset_counter[key] += row['support'] * batch_size

        merged_freq_itemsets = pd.DataFrame([
            {'itemsets': set(k), 'support': v / total_transactions}
            for k, v in itemset_counter.items()
        ])

        merged_freq_itemsets = merged_freq_itemsets.sort_values(by='support', ascending=False)
        print(merged_freq_itemsets.head(20))

        # 可选：生成关联规则（如需）
        # 需要先转化为 dummy df 格式再传入 association_rules
        # 为了演示，下面是如何构造规则（如果 itemsets 太大，可能会失败）
        try:
            rules = association_rules(merged_freq_itemsets, metric="confidence", min_threshold=0.5)
            rules_sorted = rules.sort_values(by=['confidence', 'lift'], ascending=False)
            print(rules_sorted.head(10))
        except Exception as e:
            print("生成关联规则失败，原因：", e)

        output_path = os.path.join(project_root, '../data/Shi/merged_frequent_itemsets.json')
        merged_freq_itemsets.to_json(output_path, orient='records', force_ascii=False, indent=2)
        print(f"已保存频繁项集到：{output_path}")

        rules_output_path = os.path.join(project_root, '../data/Shi/association_rules.csv')
        rules_sorted.to_csv(rules_output_path, index=False, encoding='utf-8-sig')