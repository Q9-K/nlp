import os
import requests
from bs4 import BeautifulSoup
import argparse
from fake_useragent import UserAgent
from urllib.parse import urljoin, urlparse
import argparse

urls = [
    ['http://www.xinhuanet.com/',
    'https://english.news.cn/home.htm'
    ],
    ['https://www.people.com.cn/',
    'https://en.people.cn/',
    ]
]

ALLOWED_DOMAINS = ["news.cn", "people.com.cn", "xinhuanet.com"]


headers = {
    'User-Agent': UserAgent().chrome
}

MAX_PAGES = 10000

def is_valid_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ['http', 'https']:
        return False
    if not any(parsed.netloc.endswith(domain) for domain in ALLOWED_DOMAINS):
        return False
    return True

def fetch_url(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        # print(f"Fetched {url} with status code {response.status_code}")
        return response.text
    except requests.RequestException as e:
        # print(f"Error fetching {url}: {e}")
        return None


def is_html(url):
    parsed = urlparse(url)
    if not parsed.scheme in ['http', 'https']:
        return False
    path = parsed.path.lower()
    return path.endswith('.htm') or path.endswith('.html') or path == '' or path.endswith('/')

def process_content(content, url):
    soup = BeautifulSoup(content, 'lxml')
    title = soup.title.string if soup.title else "No Title"
    text = ''
    new_urls = []
    for link in soup.find_all('a', href=True):
        full_url = urljoin(url, link['href'])
        if is_html(full_url) and is_valid_url(full_url):
            new_urls.append(full_url)
    main_content = soup.find('div', class_='rm_txt_con') or soup.find('div', class_='d2txtCon') or soup.find('div', id='detail')
    for p in main_content.find_all('p') if main_content else []:
        text += p.get_text(strip=True) + '\n'
    
    return title, text, new_urls
    
if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    parser = argparse.ArgumentParser(description="Web crawler for news websites")
    parser.add_argument('--website', type=int, choices=[0, 1], help="0 for xinhuanet, 1 for people.com.cn")
    args = parser.parse_args()
    website = args.website == 0 and 'xinhuanet' or 'people'
    crawled_urls = set()
    queue = list(urls[args.website])
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
                print(f"Crawled {current_url}, ({page_count}): {current_url}", end='\r')
                with open(filename, 'a+', encoding='utf-8') as f:
                    f.write(f"{title}\n\n")
                    f.write(text + "\n\n")