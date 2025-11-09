import os
import requests
from bs4 import BeautifulSoup
import argparse
from fake_useragent import UserAgent
from urllib.parse import urljoin, urlparse

urls = [
    'https://www.news.cn/?f=pad',
    'https://english.news.cn/home.htm',
    'https://www.people.com.cn/',
    'https://en.people.cn/',
    ]


headers = {
    'User-Agent': UserAgent().chrome
}

def fetch_url(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        print(f"Fetched {url} with status code {response.status_code}")
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def process_content(content, index = 0):
    soup = BeautifulSoup(content, 'lxml')
    title = soup.title.string if soup.title else "No Title"
    text = None
    new_urls = []
    for link in soup.find_all('a', href=True):
        full_url = urljoin(urls[index], link['href'])
        parsed = urlparse(full_url)
        if parsed.scheme in ['http', 'https']:
            new_urls.append(full_url)
    if index == 0:
        main_content = soup.find('div', id='detail')
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
    elif index == 1:
        main_content = soup.find('div', id='detail')
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
    elif index == 2:
        main_content = soup.find('div', class_='rm_txt_con')
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
    elif index == 3:
        main_content = soup.find('div', class_='d2txtCon')
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
    return title, text, new_urls
    
if __name__ == "__main__":
    for index, url in enumerate(urls):
        MAX_PAGES = 1000
        crawled_urls = set()
        queue = [url]
        page_count = 0
        filename = f'data/{index}.txt'
        while queue and page_count < MAX_PAGES:
            current_url = queue.pop(0)
            if current_url in crawled_urls:
                continue
            content = fetch_url(current_url)
            if content:
                crawled_urls.add(current_url)
                title, text, new_urls = process_content(content, index)
                queue.extend(new_urls)
                if text:
                    page_count += 1
                    with open(filename, 'a+', encoding='utf-8') as f:
                        f.write(f"{title}\n\n")
                        f.write(text)