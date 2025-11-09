import os
import requests
from bs4 import BeautifulSoup
import time
from fake_useragent import UserAgent
from urllib.parse import urljoin, urlparse
from utils.extract import extract_chinese, extract_english
import chardet
import html2text


# --- 配置参数 ---
START_URL = "https://www.news.cn/" # "https://www.people.com.cn/", "https://www.news.cn/"
MAX_PAGES_TO_CRAWL = 10000
CRAWL_DELAY_SECONDS = 0
ALLOWED_DOMAINS = ["news.cn"] # "people.com.cn", "news.cn"

lists = [('https://www.news.cn', '')]

ua = UserAgent()
HEADERS = {
    'User-Agent': ua.chrome
}

crawled_urls = set()
to_crawl_queue = [START_URL]
page_count = 0

h = html2text.HTML2Text()
h.ignore_links = True

def is_valid(url):
    """
    检查URL是否有效，是否属于允许的域名，是否是人民网的链接。
    """
    parsed = urlparse(url)
    # 排除非 HTTP/HTTPS 协议的链接
    if parsed.scheme not in ['http', 'https']:
        return False
    # 限制在指定的域名内
    if not any(parsed.netloc.endswith(domain) for domain in ALLOWED_DOMAINS):
        return False
    return True

def crawl(url):
    global page_count
    save_path = "data/news/" # "data/people/", "data/news/"

    
    if url in crawled_urls:
        return []

    # print(f"[{page_count + 1}/{MAX_PAGES_TO_CRAWL}] 正在爬取: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=100)
        if response.status_code == 200:
            encoding = chardet.detect(response.content)['encoding']
            response.encoding = encoding
            
            soup = BeautifulSoup(response.text, 'lxml')
            title = soup.title.string if soup.title else "无标题"
            print(f"[{page_count + 1}/{MAX_PAGES_TO_CRAWL}] 爬取成功: {url}")
            print(f"  -> 页面标题: {title}")
            print(f"  -> 内容长度: {len(response.text)} 字节")
            if not os.path.exists(os.path.join(save_path, 'chinese')):
                os.makedirs(os.path.join(save_path, 'chinese'))
            if not os.path.exists(os.path.join(save_path, 'english')):
                os.makedirs(os.path.join(save_path, 'english'))
            file_path1 = os.path.join(os.path.join(save_path, 'chinese'), f"page_{page_count + 1}.md")
            file_path2 = os.path.join(os.path.join(save_path, 'english'), f"page_{page_count + 1}.md")
            text = soup.get_text(separator='\n', strip=True)
            with open(file_path1, "w", encoding='utf-8') as f:
                text = h.handle(text)
                text = extract_chinese(text)
                f.write(f"# {extract_chinese(title)}\n\n")
                f.write(text)
            with open(file_path2, "w", encoding='utf-8') as f:
                text = h.handle(text)
                text = extract_english(text)
                f.write(f"# {extract_english(title)}\n\n")
                f.write(text)
            
            crawled_urls.add(url)
            page_count += 1

            found_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                if is_valid(absolute_url):
                    found_links.append(absolute_url)
            
            return found_links
        else:
            print(f"  -> 状态码错误: {response.status_code}")
            return []

    except requests.RequestException as e:
        print(f"  -> 请求失败: {e}")
        return []
    except Exception as e:
        print(f"  -> 发生错误: {e}")
        return []

print(f"爬虫启动，目标: {START_URL}，限制页面数: {MAX_PAGES_TO_CRAWL}")
print("-" * 40)

while to_crawl_queue and page_count < MAX_PAGES_TO_CRAWL:
    current_url = to_crawl_queue.pop(0)
    
    if current_url in crawled_urls:
        continue
    new_links = crawl(current_url)
    
    for link in new_links:
        if link not in crawled_urls and link not in to_crawl_queue:
            to_crawl_queue.append(link)

    if page_count < MAX_PAGES_TO_CRAWL:
        time.sleep(CRAWL_DELAY_SECONDS)

print("-" * 40)
print(f"爬取结束。总共爬取了 {page_count} 个页面。")