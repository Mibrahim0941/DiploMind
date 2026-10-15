from pymongo import MongoClient
import json
from pathlib import Path
import hashlib

client = MongoClient("mongodb://localhost:27017")
db = client["irlit"]
collection = db["articles"]

LOADED_FILES_LOG = Path("loaded_files.json")

def load_loaded_files():
    if LOADED_FILES_LOG.exists():
        return set(json.loads(LOADED_FILES_LOG.read_text()))
    return set()

def save_loaded_files(loaded):
    LOADED_FILES_LOG.write_text(json.dumps(list(loaded)))

def make_id(link):
    return hashlib.sha256(link.encode()).hexdigest()[:24]

scraped_dir = Path("scraped")
all_files = list(scraped_dir.glob("*.jsonl"))
loaded_files = load_loaded_files()
new_files = [f for f in all_files if f.name not in loaded_files]

print(f"Found {len(all_files)} files total, {len(new_files)} new since last load")

inserted = 0
updated = 0
for file in new_files:
    with open(file, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            doc_id = make_id(record["link"])
            result = collection.replace_one({"_id": doc_id}, {**record, "_id": doc_id}, upsert=True)
            if result.upserted_id:
                inserted += 1
            else:
                updated += 1
    loaded_files.add(file.name)

save_loaded_files(loaded_files)
print(f"Inserted: {inserted}, Updated: {updated}")
print(f"Total documents in collection: {collection.count_documents({})}")