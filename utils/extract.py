import re
def extract_chinese(text):
    return ''.join(re.findall(r'[\u4e00-\u9fff]', text))

def extract_english(text):
    return ' '.join(re.findall(r'[a-zA-Z]+', text))