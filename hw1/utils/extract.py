import re
import nltk
import jieba

# nltk.download()
ignore_words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'mg', 'pagebreak', 'nbsp', 'quot', 'lt', 'gt', 'amp']
def split_chinese_text(text):
    stop_words = set(nltk.corpus.stopwords.words('chinese'))
    tokens = jieba.cut(text, cut_all=False)
    return [word for word in tokens if word not in stop_words]

def extract_chinese(text):
    return split_chinese_text(''.join(re.findall(r'[\u4e00-\u9fff]', text)))

def extract_english_word(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = re.findall(r'[a-zA-Z]+', text)
    return [word.lower() for word in words if word.lower() not in stop_words and word.lower() not in ignore_words \
            and len(word) >= 5]

def extract_english_char(text):
    return re.findall(r'[a-zA-Z]', text)

if __name__ == "__main__":
    sample_text = "https://jiaotong.baidu.com/cms/reports/traffic/2021/index.html"
    print("Extracted Chinese:", extract_chinese(sample_text))
    print("Extracted English Words:", extract_english_word(sample_text))
    print("Extracted English Characters:", extract_english_char(sample_text))