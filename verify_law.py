import os
from utils.split import split_text
from utils.extract import extract_chinese
from utils.count import count_words
import matplotlib.pyplot as plt
import pandas as pd

MAX_COUNT = 10000
if __name__ == "__main__":
    contents = ''
    path = 'data/people' # 'data/people', 'data/news'
    items = os.listdir(path)
    count = 0
    for item in items:
        if item.endswith('.md'):
            count += 1
            with open(os.path.join(path, item), 'r', encoding='utf-8') as f:
                content = extract_chinese(f.read())
                contents += content + '\n'
        if count >= MAX_COUNT:
            break
    print(f"总长度: {len(contents)} 字符")
    word_counter, total_count = count_words(split_text(contents))
    sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
    ranked_words = pd.DataFrame(sorted_words, columns=['word', 'actual_freq'])
    ranked_words['rank'] = range(1, len(ranked_words)+1)
    top_words = ranked_words.head(200)
    f1 = top_words['actual_freq'].iloc[0]
    top_words['theoretical_freq'] = f1 / top_words['rank']

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文（Windows）
    # plt.figure(figsize=(10, 6))

    print("\n前10个高频词的排名、实际频率与理论频率：")
    result_df = top_words[['rank', 'word', 'actual_freq', 'theoretical_freq']].head(10)
    print(result_df.round(2))