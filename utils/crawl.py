import asyncio
from crawl4ai import *
from harvesttext import HarvestText
from scrapy.linkextractors import LinkExtractor
import re
from urlmaker import URL_REGEX

async def get_content(url: str) -> str:
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        # print(result.markdown)
        text = result.markdown
        ht = HarvestText()
        text = ht.clean_text(text)
        return text

if __name__ == "__main__":
    text = asyncio.run(get_content('https://blog.csdn.net/2301_81073317/article/details/153976089?spm=1000.2115.3001.10524'))
    print(text)
    with open('test.txt', 'w', encoding='utf-8') as f:
        f.write(text)