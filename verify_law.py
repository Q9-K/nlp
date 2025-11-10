import os
from utils.extract import extract_chinese, extract_english_word, extract_english_char
from utils.count import count_words
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

count_points = [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file', type=str, help='file path.')
    # argparser.add_argument('--type', type=int, choices=[0, 1, 2], help='Type of tokenization: char, word, or 汉字.')
    args = argparser.parse_args()
    file_path = args.file
    extract_func = extract_english_word
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.read()
    print(f"总长度: {len(contents)} 字符")
    for count in count_points:
        if len(contents) < count:
            break
        tokens = extract_func(contents[:count])
        word_counter, total_count = count_words(tokens)
        sorted_words = sorted(word_counter.items(), key=lambda x: x[1], reverse=True)
        ranked_words = pd.DataFrame(sorted_words, columns=['word', 'actual_freq'])
        ranked_words['rank'] = range(1, len(ranked_words)+1)
        f1 = ranked_words['actual_freq'].iloc[0]
        ranked_words['theoretical_freq'] = f1 / ranked_words['rank']

        print(f"\n{count}: 前10个高频词的排名、实际频率与理论频率：")
        result_df = ranked_words[['rank', 'word', 'actual_freq', 'theoretical_freq']].head(20)
        print(result_df.round(2))

        plot_n = min(200, len(ranked_words))

        plot_data = ranked_words.head(plot_n)

        # 创建画布（设置合适大小）
        plt.figure(figsize=(10, 6))

        # 绘制实际频率曲线（蓝色实线，带散点）
        plt.plot(plot_data['rank'], plot_data['actual_freq'], 
                 color='#2E86AB', marker='o', markersize=4, linewidth=2,
                 label=f'实际频率')

        # 绘制理论频率曲线（红色虚线，带散点）
        plt.plot(plot_data['rank'], plot_data['theoretical_freq'], 
                 color='#A23B72', marker='s', markersize=3, linewidth=2, linestyle='--',
                 label=f'理论频率（齐夫定律：f(r) = {f1:.0f}/r）')

        # 设置双对数坐标（核心：齐夫定律在对数坐标下呈线性）
        plt.xscale('log')
        plt.yscale('log')

        # 设置坐标轴标签和标题
        plt.xlabel('排名（log尺度）', fontsize=12)
        plt.ylabel('频率（log尺度）', fontsize=12)
        plt.title(f'齐夫定律验证： 频率-排名关系', fontsize=14, pad=20)

        # 添加网格（对数坐标网格更易观察趋势）
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # 添加图例（调整位置避免遮挡）
        plt.legend(loc='upper right', fontsize=10)

        # 调整布局（防止标签被截断）
        plt.tight_layout()
        save_dir = f'results/zipf_law/{file_path.split(os.sep)[-1].split(".")[0]}'
        # print(file_path.split(os.sep)[-1])
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{count}.png', dpi=300)