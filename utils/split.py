import jieba
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
def split_text(text):
    stop_words = set(nltk.corpus.stopwords.words('chinese'))
    tokens = jieba.cut(text, cut_all=False)
    return [word for word in tokens if word not in stop_words]


if __name__ == "__main__":
    sample_text = "我是国科大的学生"
    print(split_text(sample_text))