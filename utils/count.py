from collections import Counter


def count_words(word_list):
    total = len(word_list)
    return Counter(word_list), total

if __name__ == "__main__":
    sample_words = ["我", "来到", "北京", "清华大学", "我", "爱", "北京"]
    word_count, total_count = count_words(sample_words)
    print(word_count['我'])
    print(total_count)
    
