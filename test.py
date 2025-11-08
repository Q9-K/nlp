from urllib.parse import urlparse

url = '/test/1'
parsed = urlparse(url)
print(parsed)  # 输出: ParseResult(scheme='', netloc='', path='/test/1', params='', query='', fragment='')