from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
collection = client["irlit"]["articles"]

print("Total documents:", collection.count_documents({}))
print("By source:", {s: collection.count_documents({"source": s}) for s in ["Dawn", "BBC", "ISSI"]})
print()

for doc in collection.find().limit(5):
    print(doc["source"], "|", doc["content_type"], "|", doc["title"])
    print(" ", doc["text"][:150].replace("\n", " "), "...")
    print()