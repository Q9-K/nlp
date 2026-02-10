"""
Subword Tokenization Comparison: BPE, WordPiece, Unigram
对 news-commentary-v6.en 进行切分，分别设置词表规模为 1000、3000 和 5000
统计计算切分后的数据压缩率，对比三种方法
"""

import os
import sentencepiece as spm
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace


def read_data(file_path):
    """读取数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def train_sentencepiece_model(input_file, model_prefix, vocab_size, model_type='bpe'):
    """
    训练 SentencePiece 模型 (BPE 或 Unigram)
    Args:
        input_file: 输入文件路径
        model_prefix: 模型前缀名
        vocab_size: 词表大小
        model_type: 模型类型 ('bpe' 或 'unigram')
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        pad_id=3
    )
    print(f"模型 {model_prefix} ({model_type}) 训练完成，词表大小: {vocab_size}")


def train_wordpiece_model(input_file, model_path, vocab_size):
    """
    训练 WordPiece 模型
    Args:
        input_file: 输入文件路径
        model_path: 模型保存路径
        vocab_size: 词表大小
    """
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    tokenizer.train([input_file], trainer)
    tokenizer.save(model_path)
    print(f"WordPiece 模型训练完成，词表大小: {vocab_size}")


def load_model(model_path):
    """加载 BPE 模型"""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def load_sentencepiece_model(model_path):
    """加载 SentencePiece 模型"""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def load_wordpiece_model(model_path):
    """加载 WordPiece 模型"""
    return Tokenizer.from_file(model_path)


def tokenize_with_sentencepiece(sp, input_file):
    """
    使用 SentencePiece 模型对文件进行切分
    Args:
        sp: SentencePiece 处理器
        input_file: 输入文件路径
    Returns:
        tokens: 切分后的 token 列表
    """
    tokens = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pieces = sp.encode_as_pieces(line)
                tokens.extend(pieces)
    return tokens


def tokenize_with_wordpiece(tokenizer, input_file):
    """
    使用 WordPiece 模型对文件进行切分
    Args:
        tokenizer: WordPiece tokenizer
        input_file: 输入文件路径
    Returns:
        tokens: 切分后的 token 列表
    """
    tokens = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                output = tokenizer.encode(line)
                tokens.extend(output.tokens)
    return tokens


def calculate_compression_rate(original_file, tokens):
    """
    计算数据压缩率
    压缩率的几种计算方式：
    1. 字符数 / token数 - 平均每个token代表多少字符
    2. 原始词数 / token数 - 相对于空格分词的压缩比
    """
    # 读取原始文件
    with open(original_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 原始统计
    char_count = len(text)
    word_count = len(text.split())
    
    # 切分后统计
    token_count = len(tokens)
    
    # 计算压缩率
    char_per_token = char_count / token_count  # 每个token平均字符数
    word_token_ratio = word_count / token_count  # 词/token比率
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'token_count': token_count,
        'char_per_token': char_per_token,
        'word_token_ratio': word_token_ratio,
        'compression_rate': char_count / token_count  # 字符压缩率
    }


def analyze_vocabulary_sp(sp, tokens):
    """分析 SentencePiece 词表使用情况"""
    token_freq = Counter(tokens)
    vocab_size = sp.get_piece_size()
    unique_tokens = len(token_freq)
    top_tokens = token_freq.most_common(10)
    
    return {
        'vocab_size': vocab_size,
        'unique_tokens_used': unique_tokens,
        'vocab_coverage': unique_tokens / vocab_size * 100,
        'top_tokens': top_tokens
    }


def analyze_vocabulary_wp(tokenizer, tokens):
    """分析 WordPiece 词表使用情况"""
    token_freq = Counter(tokens)
    vocab_size = tokenizer.get_vocab_size()
    unique_tokens = len(token_freq)
    top_tokens = token_freq.most_common(10)
    
    return {
        'vocab_size': vocab_size,
        'unique_tokens_used': unique_tokens,
        'vocab_coverage': unique_tokens / vocab_size * 100,
        'top_tokens': top_tokens
    }


def run_experiment(data_file, method, vocab_size, all_results):
    """
    运行单个实验
    Args:
        data_file: 数据文件路径
        method: 方法名称 ('bpe', 'wordpiece', 'unigram')
        vocab_size: 词表大小
        all_results: 结果字典
    """
    print(f"\n--- {method.upper()} (vocab_size={vocab_size}) ---")
    
    if method in ['bpe', 'unigram']:
        # SentencePiece 方法
        model_prefix = f"{method}_vocab{vocab_size}"
        model_path = f"{model_prefix}.model"
        
        if not os.path.exists(model_path):
            print(f"正在训练 {method.upper()} 模型...")
            train_sentencepiece_model(data_file, model_prefix, vocab_size, model_type=method)
        else:
            print(f"加载已存在的模型: {model_path}")
        
        sp = load_sentencepiece_model(model_path)
        print("正在切分数据...")
        tokens = tokenize_with_sentencepiece(sp, data_file)
        vocab_stats = analyze_vocabulary_sp(sp, tokens)
        
    else:  # wordpiece
        model_path = f"wordpiece_vocab{vocab_size}.json"
        
        if not os.path.exists(model_path):
            print(f"正在训练 WordPiece 模型...")
            train_wordpiece_model(data_file, model_path, vocab_size)
        else:
            print(f"加载已存在的模型: {model_path}")
        
        tokenizer = load_wordpiece_model(model_path)
        print("正在切分数据...")
        tokens = tokenize_with_wordpiece(tokenizer, data_file)
        vocab_stats = analyze_vocabulary_wp(tokenizer, tokens)
    
    # 计算压缩率
    compression_stats = calculate_compression_rate(data_file, tokens)
    
    # 保存结果
    key = (method, vocab_size)
    all_results[key] = {
        'compression': compression_stats,
        'vocabulary': vocab_stats
    }
    
    # 打印结果
    print(f"切分后 token 数: {compression_stats['token_count']:,}")
    print(f"字符压缩率: {compression_stats['compression_rate']:.2f}")
    print(f"词/token 比率: {compression_stats['word_token_ratio']:.4f}")
    print(f"词表覆盖率: {vocab_stats['vocab_coverage']:.2f}%")
    
    return compression_stats, vocab_stats


def main():
    # 文件路径
    data_file = 'news-commentary-v6.en'
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 文件 {data_file} 不存在")
        return
    
    # 方法列表
    methods = ['bpe', 'wordpiece', 'unigram']
    
    # 词表规模设置
    vocab_sizes = [1000, 3000, 5000]
    
    all_results = {}
    
    print("=" * 70)
    print("Subword 切分方法对比实验")
    print("方法: BPE, WordPiece, Unigram")
    print("词表规模: 1000, 3000, 5000")
    print("=" * 70)
    
    # 获取原始文件统计
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    char_count = len(text)
    word_count = len(text.split())
    print(f"\n原始数据统计:")
    print(f"  字符数: {char_count:,}")
    print(f"  词数(空格分词): {word_count:,}")
    
    # 运行所有实验
    for method in methods:
        print(f"\n{'='*70}")
        print(f"方法: {method.upper()}")
        print("=" * 70)
        
        for vocab_size in vocab_sizes:
            run_experiment(data_file, method, vocab_size, all_results)
    
    # 打印对比总结
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    # 按词表规模分组显示
    for vocab_size in vocab_sizes:
        print(f"\n--- 词表规模: {vocab_size} ---")
        print(f"{'方法':<12} {'Token数':<15} {'字符压缩率':<15} {'词/Token比':<12}")
        print("-" * 55)
        
        for method in methods:
            key = (method, vocab_size)
            stats = all_results[key]['compression']
            print(f"{method:<12} {stats['token_count']:<15,} {stats['compression_rate']:<15.2f} {stats['word_token_ratio']:<12.4f}")
    
    # 综合对比表
    print("\n" + "=" * 70)
    print("综合对比表")
    print("=" * 70)
    print(f"\n{'方法':<12} {'词表':<8} {'Token数':<12} {'压缩率':<10} {'词/Token比':<12}")
    print("-" * 60)
    
    for vocab_size in vocab_sizes:
        for method in methods:
            key = (method, vocab_size)
            stats = all_results[key]['compression']
            print(f"{method:<12} {vocab_size:<8} {stats['token_count']:<12,} {stats['compression_rate']:<10.2f} {stats['word_token_ratio']:<12.4f}")
    
    print("\n" + "=" * 70)
    print("说明")
    print("=" * 70)
    print("- 字符压缩率: 每个 token 平均代表的字符数，越高表示压缩效果越好")
    print("- 词/Token比: 原始词数与 token 数的比值，越接近1表示切分粒度越接近词级别")
    print("- BPE: 基于频率的字节对编码，迭代合并最频繁的字符对")
    print("- WordPiece: 类似BPE，但基于似然值选择合并")
    print("- Unigram: 基于概率的子词模型，从大词表开始逐步裁剪")


if __name__ == "__main__":
    main()
