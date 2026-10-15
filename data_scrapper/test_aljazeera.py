import feedparser

feed = feedparser.parse("https://www.aljazeera.com/xml/rss/all.xml")
print("bozo (parse failed?):", feed.bozo)
print("bozo_exception:", feed.get("bozo_exception"))
print("status:", feed.get("status", "no response"))