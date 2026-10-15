"""
IRLit — Dawn scraper (RSS discovery + full-text extraction + permissive category filter)
pip install feedparser beautifulsoup4
"""

import feedparser
from bs4 import BeautifulSoup
import json
from pathlib import Path
from datetime import datetime, timezone

FEED_URL = "https://www.dawn.com/feeds/home"
SKIP_CATEGORIES = {"Sport", "Entertainment"}  # only the obviously-irrelevant ones

OUTPUT_FILE = Path("dawn_articles.jsonl")
CATEGORY_LOG = Path("dawn_categories_log.jsonl")


def log_category(category, title, link, skipped):
    with open(CATEGORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "category": category,
            "title": title,
            "link": link,
            "skipped": skipped,
        }, ensure_ascii=False) + "\n")


def extract_text(entry):
    raw_html = entry.content[0].value
    soup = BeautifulSoup(raw_html, "html.parser")
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    return "\n\n".join(p for p in paragraphs if len(p) > 40)


def run():
    feed = feedparser.parse(FEED_URL)
    print(f"Feed title: {feed.feed.title}")
    print(f"Entries found: {len(feed.entries)}")

    articles = []
    for entry in feed.entries:
        category = entry.get("tags", [{}])[0].get("term", "n/a")
        skip = category in SKIP_CATEGORIES
        log_category(category, entry.title, entry.link, skip)

        if skip:
            continue

        text = extract_text(entry)
        if len(text) < 200:
            continue  # extraction likely failed, don't save junk

        articles.append({
            "source": "Dawn",
            "category": category,
            "title": entry.title,
            "link": entry.link,
            "published": entry.get("published", ""),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "text": text,
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"Saved {len(articles)} articles -> {OUTPUT_FILE}")
    print(f"Category log (every category, skipped or not) -> {CATEGORY_LOG}")


if __name__ == "__main__":
    run()