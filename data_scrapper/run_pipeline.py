"""Runs the full pipeline: scrape all sources, then load into Mongo."""
import combined_scrapper as scraper

print("=== Scraping ===")
scraper.scrape_dawn()
scraper.scrape_bbc()
scraper.scrape_issi()

print("\n=== Loading into Mongo ===")
import load_to_mongo  # executes on import, same as running it directly