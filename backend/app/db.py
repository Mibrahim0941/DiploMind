import os

from pymongo import ASCENDING, MongoClient

client = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.environ.get("MONGO_DB", "diplomind")]

users = db["users"]
onboarding_questions = db["onboarding_questions"]
onboarding_sessions = db["onboarding_sessions"]


def init_indexes():
    # One account per email (use case 1). MongoDB rejects a duplicate insert.
    users.create_index([("email", ASCENDING)], unique=True)
    onboarding_sessions.create_index([("user_id", ASCENDING)])