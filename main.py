import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import html2text
import asyncio
from crawl4ai import *
from utils.extract import extract_chinese

# --- 配置参数 ---
START_URL = "https://www.news.cn/"
MAX_PAGES_TO_CRAWL = 10000
CRAWL_DELAY_SECONDS = 0
ALLOWED_DOMAINS = ["news.cn"]


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36'
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
    
    if url in crawled_urls:
        return []

    print(f"[{page_count + 1}/{MAX_PAGES_TO_CRAWL}] 正在爬取: {url}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        # 检查响应状态码，只处理成功的页面
        if response.status_code == 200:
            response.encoding = 'UTF-8'
            
            # **核心内容提取** (您可以根据需要修改这部分)
            soup = BeautifulSoup(response.text, 'html.parser')
            # 示例: 提取标题和正文部分内容的长度
            title = soup.title.string if soup.title else "无标题"
            print(f"  -> 页面标题: {title}")
            print(f"  -> 内容长度: {len(response.text)} 字节")
            with open(f"data/news/page_{page_count + 1}.md", "w", encoding='utf-8') as f:
                text = h.handle(response.text)
                text = extract_chinese(text)
                f.write(f"# {extract_chinese(title)}\n\n")
                f.write(text)
            
            # 将当前URL添加到已爬取集合
            crawled_urls.add(url)
            page_count += 1
            
            # **链接提取**
            found_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                # 将相对链接转换为绝对链接
                absolute_url = urljoin(url, href)
                # 检查链接是否合法且属于目标域名
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

# --- 主爬取循环 ---
print(f"爬虫启动，目标: {START_URL}，限制页面数: {MAX_PAGES_TO_CRAWL}")
print("-" * 40)

while to_crawl_queue and page_count < MAX_PAGES_TO_CRAWL:
    current_url = to_crawl_queue.pop(0)  # 取出队列中的第一个链接 (BFS)
    
    # 检查是否已爬取或超出限制
    if current_url in crawled_urls:
        continue

    # 执行爬取操作
    new_links = crawl(current_url)
    
    # 将新链接加入待爬取队列，并去重
    for link in new_links:
        if link not in crawled_urls and link not in to_crawl_queue:
            to_crawl_queue.append(link)

    # **道德延迟**：在每次请求后暂停
    if page_count < MAX_PAGES_TO_CRAWL:
        time.sleep(CRAWL_DELAY_SECONDS)

print("-" * 40)
print(f"爬取结束。总共爬取了 {page_count} 个页面。")