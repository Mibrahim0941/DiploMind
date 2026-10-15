"""
IRLit — combined scraper: Dawn (RSS w/ full content), BBC (RSS + page fetch),
ISSI (HTML discovery + PDF extraction). Each source outputs the same JSONL shape.

pip install requests beautifulsoup4 feedparser pdfplumber
"""

import requests
from bs4 import BeautifulSoup
import feedparser
import pdfplumber
import io
import json
import time
from pathlib import Path
from datetime import datetime, timezone

OUTPUT_DIR = Path("scraped")
OUTPUT_DIR.mkdir(exist_ok=True)
CATEGORY_LOG = Path("category_log.jsonl")
SEEN_FILE = Path("seen_links.json")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def load_seen():
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()


def save_seen(seen):
    SEEN_FILE.write_text(json.dumps(list(seen)))


def log_category(source, category, title, link, skipped):
    with open(CATEGORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "source": source, "category": category, "title": title,
            "link": link, "skipped": skipped,
        }, ensure_ascii=False) + "\n")


def save_records(source, records):
    if not records:
        print(f"  {source}: nothing new")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = OUTPUT_DIR / f"{source.lower()}_{stamp}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {source}: saved {len(records)} -> {out_file}")


# ---------------- DAWN — validated end-to-end already ----------------

def scrape_dawn():
    print("Scraping Dawn...")
    SKIP_CATEGORIES = {"Sport", "Entertainment"}
    seen = load_seen()
    feed = feedparser.parse("https://www.dawn.com/feeds/home")

    if feed.bozo:
        print(f"  WARNING: Dawn fetch failed - {feed.get('bozo_exception')}")
        return
    if len(feed.entries) == 0:
        print("  WARNING: Dawn feed returned zero entries (possible block/outage)")
        return

    records = []
    for entry in feed.entries:
        if entry.link in seen:
            continue  # already scraped in a previous run

        category = entry.get("tags", [{}])[0].get("term", "n/a")
        skip = category in SKIP_CATEGORIES
        log_category("Dawn", category, entry.title, entry.link, skip)
        if skip:
            continue

        soup = BeautifulSoup(entry.content[0].value, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        text = "\n\n".join(p for p in paragraphs if len(p) > 40)
        if len(text) < 200:
            continue

        records.append({
            "source": "Dawn", "content_type": "current_affairs", "category": category,
            "title": entry.title, "link": entry.link, "published": entry.get("published", ""),
            "scraped_at": datetime.now(timezone.utc).isoformat(), "text": text,
        })
        seen.add(entry.link)

    save_seen(seen)
    save_records("Dawn", records)


# ---------------- BBC — feed + extraction both validated separately ----------------

def scrape_bbc():
    print("Scraping BBC World...")
    seen = load_seen()
    feed = feedparser.parse("https://feeds.bbci.co.uk/news/world/rss.xml")

    records = []
    for entry in feed.entries:
        if "/videos/" in entry.link:
            continue  # video content, no article text to extract
        if entry.link in seen:
            continue  # already scraped in a previous run — skip before fetching

        try:
            resp = requests.get(entry.link, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            text = "\n\n".join(p for p in paragraphs if len(p) > 40)
        except Exception as e:
            print(f"  skip {entry.link}: {e}")
            continue

        if len(text) < 200:
            continue

        seen.add(entry.link)  # only mark seen once it actually succeeded
        records.append({
            "source": "BBC", "content_type": "current_affairs", "category": "World",
            "title": entry.title, "link": entry.link, "published": entry.get("published", ""),
            "scraped_at": datetime.now(timezone.utc).isoformat(), "text": text,
        })

    save_seen(seen)
    save_records("BBC", records)


# ---------------- ISSI — one layer here is NOT yet verified, see note below ----------------

def scrape_issi(issue_limit=None):
    """
    Both discovery layers verified against real output. Additions after the
    connection-reset crash: sleep between requests (likely cause), try/except
    around each fetch (skip one bad article, don't kill the run), and a
    try/finally so whatever's been collected gets saved even if something
    unexpected still goes wrong partway through.
    """
    print("Scraping ISSI...")
    seen = load_seen()
    category_url = "https://issi.org.pk/category/issi-publications/strategic-studies/"
    resp = requests.get(category_url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")

    container = soup.find("div", class_="td-ss-main-content")
    issue_links = [h.find("a")["href"] for h in container.find_all("h3", class_="entry-title")]
    if issue_limit:
        issue_links = issue_links[:issue_limit]

    records = []
    try:
        for issue_url in issue_links:
            try:
                resp = requests.get(issue_url, headers=HEADERS, timeout=15)
                soup = BeautifulSoup(resp.text, "html.parser")
            except Exception as e:
                print(f"  skip issue {issue_url}: {e}")
                continue

            content = soup.find("div", class_="td-post-content")
            table = content.find("table") if content else None
            article_links = [a["href"] for a in table.find_all("a", href=True)] if table else []
            article_links = list(dict.fromkeys(article_links))
            new_links = [a for a in article_links if a not in seen]
            print(f"  issue {issue_url} -> {len(article_links)} article links, {len(new_links)} new")

            for article_url in new_links:
                time.sleep(1.5)  # politeness delay -- likely what tripped the server before
                try:
                    resp = requests.get(article_url, headers=HEADERS, timeout=15)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    pdf_link = soup.find("a", href=lambda h: h and h.endswith(".pdf"))
                    if not pdf_link:
                        continue

                    pdf_resp = requests.get(pdf_link["href"], headers=HEADERS, timeout=30)
                    with pdfplumber.open(io.BytesIO(pdf_resp.content)) as pdf:
                        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                except Exception as e:
                    print(f"  skip article {article_url}: {e}")
                    continue

                if len(text) < 500:
                    continue

                seen.add(article_url)  # only mark seen once it actually succeeded
                title = soup.find("h1")
                records.append({
                    "source": "ISSI", "content_type": "foundational",
                    "title": title.get_text(strip=True) if title else "",
                    "link": article_url, "pdf_url": pdf_link["href"],
                    "scraped_at": datetime.now(timezone.utc).isoformat(), "text": text,
                })
            time.sleep(1)
    finally:
        save_seen(seen)
        save_records("ISSI", records)  # runs even if the loop above raised


if __name__ == "__main__":
    scrape_dawn()
    scrape_bbc()
    scrape_issi()  # verified — runs all 10 issues