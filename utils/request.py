import requests
import httpx

def fetch_url(url):
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        # "referer": "https://www.google.com/",
        # "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        print(response.status_code)
        return None
    
if __name__ == "__main__":
    url = 'https://zh.wikipedia.org/wiki'
    content = fetch_url(url)
    if content:
        print(content)
        print("Fetched content successfully.")
    else:
        print("Failed to fetch content.")