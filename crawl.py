import os
import requests
from bs4 import BeautifulSoup
import argparse
from fake_useragent import UserAgent
from urllib.parse import urljoin, urlparse
import argparse
import logging

urls = [
    'https://www.news.cn/?f=pad',
    # 'https://english.news.cn/home.htm',
    'https://www.people.com.cn/',
    # 'https://en.people.cn/',
    ]


headers = {
    'User-Agent': UserAgent().chrome
}

MAX_PAGES = 10000
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_url(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        logging.info(f"Fetched {url} with status code {response.status_code}")
        return response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

def process_content(content, url):
    soup = BeautifulSoup(content, 'lxml')
    title = soup.title.string if soup.title else "No Title"
    text = ''
    new_urls = []
    for link in soup.find_all('a', href=True):
        full_url = urljoin(url, link['href'])
        parsed = urlparse(full_url)
        if parsed.scheme in ['http', 'https']:
            new_urls.append(full_url)
    paragraphs = [p for p in soup.find_all('p') if not p.find('a')]
    for p in paragraphs:
        text += p.get_text().strip()
    
    return title, text, new_urls
    
if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    parser = argparse.ArgumentParser(description="Web crawler for news websites")
    parser.add_argument('--website', type=int, choices=[0, 1], default=0, help="0 for xinhuanet, 1 for people.com.cn")
    args = parser.parse_args()
    website = args.website == 0 and 'xinhuanet' or 'people'
    url = urls[args.website]
    crawled_urls = set()
    queue = [url]
    page_count = 0
    filename = f'data/{website}.txt'
    while queue and page_count < MAX_PAGES:
        current_url = queue.pop(0)
        if current_url in crawled_urls:
            continue
        content = fetch_url(current_url)
        if content:
            crawled_urls.add(current_url)
            title, text, new_urls = process_content(content, current_url)
            queue.extend(new_urls)
            page_count += 1
            if text:
                logging.info(f"Crawled ({page_count}): {current_url}")
                with open(filename, 'a+', encoding='utf-8') as f:
                    f.write(f"{title}\n\n")
                    f.write(text + "\n\n")