import requests
from bs4 import BeautifulSoup

url = "https://issi.org.pk/strategic-studies-vol-46-summer-2026-no-1/"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(resp.text, "html.parser")

# walk up from a link we know is a real article
target = soup.find("a", href=lambda h: h and "contested-restraint" in h)
el = target
for _ in range(6):
    el = el.parent
    if el is None:
        break
    print(el.name, el.get("class"))