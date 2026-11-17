"""Print a readable log of onboarding quiz sessions.

Run inside the backend container:
  docker compose exec backend python scripts/show_quiz_sessions.py              (last 5 sessions, all users)
  docker compose exec backend python scripts/show_quiz_sessions.py you@mail.com (last 5 sessions of one user)
"""
import os
import sys

from pymongo import DESCENDING, MongoClient

db = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017"))[os.environ.get("MONGO_DB", "diplomind")]
IDK = 4

query = {}
if len(sys.argv) > 1:
    user = db.users.find_one({"email": sys.argv[1].strip().lower()})
    if not user:
        sys.exit(f"No user with email {sys.argv[1]}")
    query["user_id"] = user["_id"]

questions = {q["_id"]: q for q in db.onboarding_questions.find({"kind": "quiz"})}
users = {u["_id"]: u for u in db.users.find({}, {"email": 1, "register": 1})}

for s in db.onboarding_sessions.find(query).sort("started_at", DESCENDING).limit(5):
    u = users.get(s["user_id"], {})
    print("=" * 100)
    print(f"{u.get('email', '?')}  |  language: {u.get('register')}  |  started {s['started_at']:%Y-%m-%d %H:%M}  "
          f"|  {'finished' if s['finished_at'] else 'NOT finished'}")
    print("-" * 100)
    for n, a in enumerate(s["answers"], 1):
        q = questions.get(a["question_id"], {})
        en = q.get("texts", {}).get("en", {})
        picked = "I don't know" if a["choice"] == IDK else en.get("options", ["?"] * 4)[a["choice"]]
        where = f"{q.get('ladder')} rung {q.get('rung')}" if q.get("ladder") else "standalone"
        b = a["belief_after"]
        print(f"{n}. [{a['question_id']}, {where}] {en.get('stem', '')[:85]}")
        print(f"   answered: {picked[:60]}  ->  {'RIGHT' if a['correct'] else 'wrong'}"
              f"   | belief easy {b[0]:.2f}  medium {b[1]:.2f}  technical {b[2]:.2f}")
    if s.get("result"):
        r = s["result"]
        print(f"RESULT: {r['depth'].upper()} (confidence {r['confidence']:.2f}) after {len(s['answers'])} questions")
print("=" * 100)