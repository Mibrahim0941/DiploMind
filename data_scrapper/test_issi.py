import requests
from bs4 import BeautifulSoup

url = "https://issi.org.pk/strategic-studies-vol-46-summer-2026-no-1/"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(resp.text, "html.parser")

content = soup.find("div", class_="td-post-content")
table = content.find("table")
article_links = [a["href"] for a in table.find_all("a", href=True)] if table else []
article_links = list(dict.fromkeys(article_links))

print("Found:", len(article_links))
for link in article_links:
    print(" ", link)