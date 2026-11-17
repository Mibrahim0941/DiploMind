"""Placement engine for the onboarding knowledge quiz (pure Python, no web code).

Keeps a belief over the three depth levels and updates it with Bayes' rule after each answer.
The next question is the allowed question with the highest expected information gain.
Ladder rules: a correct answer moves up the same topic; a wrong answer (or "I don't know")
closes that topic."""
import math
import random

LEVELS = ["easy", "medium", "technical"]
MAX_QUESTIONS = 8
MIN_QUESTIONS = 4
CONFIDENCE = 0.80              # stop once one level is this likely (after MIN_QUESTIONS)
SKIP_EASY_RUNG_BELOW = 0.20    # start new topics at rung 2 once P(easy) is below this
TIER_UNLOCK = {"Front page": 0, "Inside pages": 3, "Specialist pages": 5}  # questions asked before a tier opens
STANDALONE_FROM = MAX_QUESTIONS - 2   # standalones only for the last two questions
FIRST_PICK_TOP_K = 3           # first question: random among the 3 most informative
FIRST_MIN_P_MEDIUM = 0.60      # first question must be one most medium users get right

def new_state():
    return {"belief": [1 / 3, 1 / 3, 1 / 3], "asked": [], "ladders": {}, "current_ladder": None}


def _p(item):
    return [item["p"][L] for L in LEVELS]


def _entropy(b):
    return -sum(x * math.log(x) for x in b if x > 0)


def _posterior(b, p, correct):
    nb = [b[i] * (p[i] if correct else 1 - p[i]) for i in range(3)]
    s = sum(nb)
    return [x / s for x in nb]


def _gain(b, item):
    """Expected drop in uncertainty from asking this question."""
    p = _p(item)
    pc = sum(b[i] * p[i] for i in range(3))
    after = pc * _entropy(_posterior(b, p, True)) + (1 - pc) * _entropy(_posterior(b, p, False))
    return _entropy(b) - after


def _group(items):
    ladders, standalones = {}, []
    for it in items:
        if it.get("ladder"):
            ladders.setdefault(it["ladder"], []).append(it)
        else:
            standalones.append(it)
    for qs in ladders.values():
        qs.sort(key=lambda q: q["rung"])
    return ladders, standalones


def candidates(state, items):
    ladders, standalones = _group(items)
    n = len(state["asked"])

    # keep climbing the current topic while the user answers correctly
    cur = state["current_ladder"]
    if cur:
        nxt = state["ladders"].get(cur)
        if isinstance(nxt, int) and nxt < len(ladders[cur]):
            return [ladders[cur][nxt]]

    # otherwise: start a topic not used yet
    out = []
    for lid, qs in ladders.items():
        if lid in state["ladders"] or n < TIER_UNLOCK.get(qs[0]["tier"], 0):
            continue
        start = 1 if state["belief"][0] < SKIP_EASY_RUNG_BELOW and len(qs) > 1 else 0
        out.append(qs[start])
    if n >= STANDALONE_FROM:
        out += [q for q in standalones if q["_id"] not in state["asked"]]
    return out


def next_question(state, items, rng=random):
    cands = candidates(state, items)
    if not cands:
        return None
    if not state["asked"]:
        # friendly start: only questions most medium users answer correctly
        friendly = [q for q in cands if q["p"]["medium"] >= FIRST_MIN_P_MEDIUM]
        cands = friendly or cands
        ranked = sorted(cands, key=lambda q: _gain(state["belief"], q), reverse=True)
        return rng.choice(ranked[:FIRST_PICK_TOP_K])
    return max(cands, key=lambda q: _gain(state["belief"], q))


def record_answer(state, item, correct):
    state["belief"] = _posterior(state["belief"], _p(item), correct)
    state["asked"].append(item["_id"])
    lid = item.get("ladder")
    if lid:
        if correct:
            state["ladders"][lid] = item["rung"]      # index of the next rung
            state["current_ladder"] = lid
        else:
            state["ladders"][lid] = "closed"
            state["current_ladder"] = None
    else:
        state["current_ladder"] = None


def is_done(state, items):
    n = len(state["asked"])
    if n >= MAX_QUESTIONS:
        return True
    if n >= MIN_QUESTIONS and max(state["belief"]) >= CONFIDENCE:
        return True
    return not candidates(state, items)


def result(state):
    b = state["belief"]
    i = b.index(max(b))
    return {"depth": LEVELS[i], "confidence": round(b[i], 3),
            "belief": {L: round(x, 3) for L, x in zip(LEVELS, b)}}