import os

from pymongo import ASCENDING, MongoClient

client = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.environ.get("MONGO_DB", "diplomind")]

users = db["users"]


def init_indexes():
    # One account per email (use case 1). MongoDB rejects a duplicate insert.
    users.create_index([("email", ASCENDING)], unique=True)