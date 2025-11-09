import os
from utils.split import split_text
from utils.extract import extract_chinese
from utils.count import count_words
import math

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
    word_count, total_count = count_words(split_text(contents))
    # 统计出现的词的概率和熵
    probabilities = [count / total_count for count in word_count.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    print(f"熵: {entropy}")