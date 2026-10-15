from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
try:
    client.admin.command("ping")
    print("Connected OK")
    print("Existing databases:", client.list_database_names())
except Exception as e:
    print("Connection failed:", e)
    