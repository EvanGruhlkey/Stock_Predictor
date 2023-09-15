# Web Scraper
import requests
import json
from bs4 import BeautifulSoup
response = requests.get(
    'https://www.sofascore.com/arsenal-manchester-city/rsR#10385636',
    # you'll be blocked if you don't use some type of user agent
    headers={'User-Agent': 'Mozilla/5.0'}
)
print(response.status_code)
soup = BeautifulSoup(response.text, 'html.parser')
soup.select('g[cursor="pointer"]')

headers = {
    'authority': 'api.sofascore.com',
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    'if-none-match': 'W/"f28f77394c"',
    'origin': 'https://www.sofascore.com',
    'referer': 'https://www.sofascore.com/',
    'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
}

response = requests.get(
    'https://api.sofascore.com/api/v1/event/10385636/shotmap', headers=headers)
print(response.status_code)
headers['If-Modified-Since'] = 'Fri, 15 Sep 2023 00:00:00 GMT'
response = requests.get(
    'https://api.sofascore.com/api/v1/event/10385636/shotmap', headers=headers)
print(response.status_code)
shots = response.json()
parsed_json = json.dumps(shots, ensure_ascii=True, indent=1)
print(parsed_json)
