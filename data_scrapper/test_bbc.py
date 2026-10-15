import requests
from bs4 import BeautifulSoup

url = "https://www.bbc.co.uk/news/articles/c62vy3wl31lo"  # the Syria airstrike one, genuinely IR-relevant
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(resp.text, "html.parser")
paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
text = "\n\n".join(p for p in paragraphs if len(p) > 40)

print("STATUS:", resp.status_code)
print("LENGTH:", len(text))
print()
print(text[:800])