from utils.extract import extract_chinese, extract_english_word, extract_english_char
from utils.count import count_words
import math
import argparse
from matplotlib import pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

count_points = [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file', type=str, help='file path.')
    argparser.add_argument('--type', type=int, choices=[0, 1, 2], help='Type of tokenization: char, word, or 汉字.')
    args = argparser.parse_args()
    contents = ''
    extract_func = [extract_english_char, extract_english_word, extract_chinese][args.type]
    count_index = 0
    plot_lengths = []
    plot_entropies = []
    file_path = args.file
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.read()
    print(f"总长度: {len(contents)} 字符")
    for count in count_points:
        if len(contents) < count:
            break
        tokens = extract_func(contents[:count])
        word_count, total_count = count_words(tokens)
        # print(tokens)
        # 统计出现的词的概率和熵
        probabilities = [count / total_count for count in word_count.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities)
        print(f"熵: {entropy}")
        plot_lengths.append(len(contents[:count]))
        plot_entropies.append(entropy)
    # 绘制熵随文本长度变化的图表
    if plot_lengths and plot_entropies:
        fig, ax = plt.subplots()
        ax.plot(plot_lengths, plot_entropies, marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax.set_xlabel('累计字符长度', fontsize=12)
        ax.set_ylabel('熵值', fontsize=12)
        token_type = ['char', 'word', 'chinese'][args.type]
        ax.set_title(f'{token_type}粒度下的熵值变化曲线', fontsize=14, fontweight='bold')
        
        # 优化坐标轴格式（千位分隔符）
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # 添加网格线（便于读取数值）
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 调整布局，避免标签被截断
        plt.tight_layout()
        save_dir = f'results/entropy/{file_path.split(os.sep)[-1].split(".")[0]}'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{token_type}.png', dpi=300)