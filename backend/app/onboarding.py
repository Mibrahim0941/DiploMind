import json
import random
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import engine
from .auth import current_user
from .db import onboarding_questions, onboarding_sessions, users

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])

DATA_DIR = Path(__file__).parent / "onboarding_data"
REGISTERS = ["en", "ur", "urdish"]
REGISTER_META = {}  # prompt + register names, filled at startup
QUIZ_META = {}      # bank version + calibration date, filled at startup


# ---------- loading content from the repo into MongoDB ----------

def load_onboarding_content():
    """Copy the JSON files in onboarding_data/ into MongoDB.
    Runs at every backend start, so the database always matches the repo."""

    # language (register) samples
    with open(DATA_DIR / "register_items.json", encoding="utf-8") as f:
        reg = json.load(f)
    REGISTER_META.update(prompt=reg["prompt"], registers=reg["registers"], version=reg["version"])

    ids = []
    for i, item in enumerate(reg["items"]):
        doc = {"_id": item["id"], "kind": "register", "order": i, "texts": item["texts"],
               "version": reg["version"]}
        onboarding_questions.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        ids.append(item["id"])
    # remove register items that were deleted from the file
    onboarding_questions.delete_many({"kind": "register", "_id": {"$nin": ids}})
    print(f"Onboarding: loaded {len(ids)} register items")

    # knowledge quiz questions
    with open(DATA_DIR / "quiz_bank.json", encoding="utf-8") as f:
        quiz = json.load(f)
    QUIZ_META.update(version=quiz["version"], calibrated_on=quiz["calibrated_on"])

    qids = []
    for i, item in enumerate(quiz["items"]):
        doc = {k: v for k, v in item.items() if k != "id"}
        doc.update(_id=item["id"], kind="quiz", order=i, version=quiz["version"])
        onboarding_questions.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        qids.append(item["id"])
    # remove quiz questions that were deleted from the file
    onboarding_questions.delete_many({"kind": "quiz", "_id": {"$nin": qids}})
    print(f"Onboarding: loaded {len(qids)} quiz questions "
          f"(bank {quiz['version']}, calibrated {quiz['calibrated_on']})")


def register_items():
    return list(onboarding_questions.find({"kind": "register"}).sort("order", 1))


def item_out(doc):
    """One language sample for the frontend, with the three versions in random order
    so the position on screen does not influence the choice."""
    order = REGISTERS[:]
    random.shuffle(order)
    return {"item_id": doc["_id"], "options": [{"register": r, "text": doc["texts"][r]} for r in order]}


# ---------- endpoints ----------

@router.get("/register-items")
def get_register_items(user: dict = Depends(current_user)):
    """The first two language samples to show."""
    items = register_items()
    return {
        "prompt": REGISTER_META["prompt"],
        "registers": REGISTER_META["registers"],
        "items": [item_out(d) for d in items[:2]],
    }


class Choice(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    item_id: str
    reg: str = Field(alias="register")


class RegisterIn(BaseModel):
    choices: list[Choice]
    direct: str | None = None  # only used when all three picks differ


@router.post("/register")
def set_register(body: RegisterIn, user: dict = Depends(current_user)):
    """Send the user's picks so far.
    2 picks that agree -> register is saved.
    2 picks that differ -> returns the third sample to show.
    3 picks -> the register picked most often is saved.
    3 different picks -> asks the user directly (send again with 'direct')."""
    items = register_items()
    valid_ids = [d["_id"] for d in items]
    picks = body.choices

    if not 2 <= len(picks) <= 3:
        raise HTTPException(400, "Send 2 or 3 choices.")
    if [c.item_id for c in picks] != valid_ids[:len(picks)]:
        raise HTTPException(400, "Choices must be for the samples in the order they were shown.")
    if any(c.reg not in REGISTERS for c in picks):
        raise HTTPException(400, "Unknown register.")

    regs = [c.reg for c in picks]
    if len(regs) == 2 and regs[0] != regs[1]:
        return {"done": False, "next_item": item_out(items[2])}

    if len(set(regs)) == 3:  # all three different: ask the user directly
        if body.direct is None:
            return {"done": False, "ask_direct": True}
        if body.direct not in REGISTERS:
            raise HTTPException(400, "Unknown register.")
        register = body.direct
    else:
        register = max(set(regs), key=regs.count)

    users.update_one({"_id": user["_id"]}, {"$set": {
        "register": register,
        "onboarding.register_choices": [c.model_dump(by_alias=True) for c in picks],
        "onboarding.register_direct": body.direct,
        "onboarding.register_set_at": datetime.now(timezone.utc),
    }})
    return {"done": True, "register": register}


# ---------- knowledge quiz ----------

IDK_TEXT = {"en": "I don't know", "ur": "مجھے نہیں معلوم", "urdish": "Mujhe nahi pata"}


def quiz_items():
    return list(onboarding_questions.find({"kind": "quiz"}))


def question_out(item, register, number):
    """Question in the user's language (English if no translation yet), plus 'I don't know' as the last option.
    The correct answer is never sent."""
    lang = register if item["texts"].get(register) else "en"
    t = item["texts"][lang]
    return {"question_id": item["_id"], "number": number, "lang": lang,
            "stem": t["stem"], "options": t["options"] + [IDK_TEXT[lang]]}


@router.post("/quiz/start")
def quiz_start(user: dict = Depends(current_user)):
    if not user.get("register"):
        raise HTTPException(400, "Please choose a language first.")
    items = quiz_items()
    state = engine.new_state()
    q = engine.next_question(state, items)
    sid = onboarding_sessions.insert_one({
        "user_id": user["_id"],
        "bank_version": QUIZ_META.get("version"),
        "state": state,
        "current_question": q["_id"],
        "answers": [],
        "started_at": datetime.now(timezone.utc),
        "finished_at": None,
        "result": None,
    }).inserted_id
    return {"session_id": str(sid), "question": question_out(q, user["register"], 1)}


class AnswerIn(BaseModel):
    session_id: str
    question_id: str
    choice: int  # 0-3 = options A-D, 4 = "I don't know"


@router.post("/quiz/answer")
def quiz_answer(body: AnswerIn, user: dict = Depends(current_user)):
    try:
        sid = ObjectId(body.session_id)
    except Exception:
        raise HTTPException(400, "Invalid session.")
    s = onboarding_sessions.find_one({"_id": sid, "user_id": user["_id"]})
    if not s or s["finished_at"]:
        raise HTTPException(400, "This quiz has already finished. Please start again.")
    if body.question_id != s["current_question"]:
        raise HTTPException(400, "That is not the current question.")
    if not 0 <= body.choice <= 4:
        raise HTTPException(400, "Invalid choice.")

    items = quiz_items()
    item = next(i for i in items if i["_id"] == body.question_id)
    correct = body.choice == item["correct"]
    state = s["state"]
    engine.record_answer(state, item, correct)
    answers = s["answers"] + [{"question_id": item["_id"], "choice": body.choice, "correct": correct,
                               "belief_after": state["belief"]}]
    now = datetime.now(timezone.utc)

    if engine.is_done(state, items):
        res = engine.result(state)
        onboarding_sessions.update_one({"_id": sid}, {"$set": {
            "state": state, "answers": answers, "current_question": None, "finished_at": now, "result": res}})
        users.update_one({"_id": user["_id"]}, {"$set": {
            "depth": res["depth"],
            "onboarding_completed": True,
            "onboarding.depth_belief": res["belief"],
            "onboarding.depth_confidence": res["confidence"],
            "onboarding.quiz_session_id": sid,
            "onboarding.completed_at": now,
        }})
        return {"done": True, "depth": res["depth"], "confidence": res["confidence"]}

    q = engine.next_question(state, items)
    onboarding_sessions.update_one({"_id": sid}, {"$set": {
        "state": state, "answers": answers, "current_question": q["_id"]}})
    return {"done": False, "question": question_out(q, user["register"], len(state["asked"]) + 1)}