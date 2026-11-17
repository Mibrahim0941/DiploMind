"""
Update the p values in quiz_bank.json from the pilot study responses.

  python update_p_values.py <quiz_bank.json> <pilot_responses.xlsx>

p = chance that an easy / medium / technical user answers a question correctly.

Method
  1. Each pilot participant gets a level from their knowledge:
     IR / political science students = technical; others with at least half their form correct = medium; rest = easy.
  2. Evidence per question and level = pilot answers + the faculty reviewer's estimate (counted as
     TEACHER_WEIGHT people) where one exists.
  3. A Rasch-style model is fitted on ALL questions together:
         p(level, question) = sigmoid(ability[level] - difficulty[question])
     so questions with few answers borrow the typical gap between levels from every other question.
  4. Final p = model value blended with the question's own evidence (the model counts as MODEL_WEIGHT people),
     so questions with plenty of their own data keep their own pattern.
  5. Values are kept between 0.05 and 0.95, and easy <= medium <= technical.
The response file is always the FULL download from Google Forms, so running it again never double-counts.
"""
import json
import sys
from datetime import date

import numpy as np
import openpyxl

LEVELS = ["easy", "medium", "technical"]
TEACHER_WEIGHT = 4
MODEL_WEIGHT = 4
MEDIUM_CUTOFF = 0.5
P_MIN, P_MAX = 0.05, 0.95
Q_FIELD = "What are you studying, or what is your field of work?"


def participant_level(field, score):
    if field and "International relations" in str(field):
        return "technical"
    return "medium" if score >= MEDIUM_CUTOFF else "easy"


def read_responses(path, stem_to_item):
    """Returns per question and level: [correct, asked], plus the number of people per level."""
    counts = {it["id"]: {L: [0, 0] for L in LEVELS} for it in stem_to_item.values()}
    people = {L: 0 for L in LEVELS}
    wb = openpyxl.load_workbook(path, data_only=True)
    for ws in wb:
        rows = [r for r in ws.iter_rows(values_only=True) if any(c is not None for c in r)]
        if len(rows) < 2:
            continue
        hdr = [str(h).strip() if h else "" for h in rows[0]]
        if not any(h in stem_to_item for h in hdr):
            continue
        for r in rows[1:]:
            ans = dict(zip(hdr, r))
            marks = {}
            for col, a in ans.items():
                it = stem_to_item.get(col)
                if it:
                    correct_text = it["texts"]["en"]["options"][it["correct"]]
                    marks[it["id"]] = int(str(a).strip() == str(correct_text).strip())
            if not marks:
                continue
            L = participant_level(ans.get(Q_FIELD), sum(marks.values()) / len(marks))
            people[L] += 1
            for qid, m in marks.items():
                counts[qid][L][0] += m
                counts[qid][L][1] += 1
    return counts, people


def fit_rasch(C, N, iters=4000, lr=0.5):
    """C, N: arrays [questions x 3] of (fractional) correct and asked counts.
    Fits difficulty per question and ability per level (easy fixed at 0)."""
    q = C.shape[0]
    diff = np.zeros(q)
    ability = np.array([0.0, 1.0, 2.0])
    for _ in range(iters):
        z = ability[None, :] - diff[:, None]
        p = 1 / (1 + np.exp(-z))
        resid = C - N * p                                # gradient of the log-likelihood w.r.t. z
        g_diff = -resid.sum(1) - diff / 4.0              # weak pull of difficulties towards 0
        g_ab = resid.sum(0)
        g_ab[0] = 0.0                                    # easy ability fixed at 0
        diff += lr * g_diff / np.maximum(N.sum(1), 1)
        ability += lr * g_ab / np.maximum(N.sum(0), 1)
    return diff, ability


def make_monotonic(vals):
    vals = list(vals)
    changed = True
    while changed:
        changed = False
        for i in range(2):
            if vals[i] > vals[i + 1]:
                vals[i] = vals[i + 1] = (vals[i] + vals[i + 1]) / 2
                changed = True
    return vals


def main(bank_path, responses_path):
    with open(bank_path, encoding="utf-8") as f:
        bank = json.load(f)
    items = bank["items"]
    stem_to_item = {it["texts"]["en"]["stem"].strip(): it for it in items}
    counts, people = read_responses(responses_path, stem_to_item)

    C = np.zeros((len(items), 3))
    N = np.zeros((len(items), 3))
    for i, it in enumerate(items):
        for j, L in enumerate(LEVELS):
            c, n = counts[it["id"]][L]
            t = (it.get("teacher_p") or {}).get(L)
            if t is not None:
                c += TEACHER_WEIGHT * t
                n += TEACHER_WEIGHT
            C[i, j], N[i, j] = c, n

    diff, ability = fit_rasch(C, N)
    model = 1 / (1 + np.exp(-(ability[None, :] - diff[:, None])))

    for i, it in enumerate(items):
        vals = [(MODEL_WEIGHT * model[i, j] + C[i, j]) / (MODEL_WEIGHT + N[i, j]) for j in range(3)]
        vals = make_monotonic([min(max(v, P_MIN), P_MAX) for v in vals])
        it["p"] = {L: round(v, 3) for L, v in zip(LEVELS, vals)}
        it["pilot_n"] = {L: counts[it["id"]][L][1] for L in LEVELS}
        it["difficulty"] = round(float(diff[i]), 3)

    bank["calibrated_on"] = date.today().isoformat()
    bank["pilot_participants"] = people
    bank["level_ability"] = {L: round(float(a), 3) for L, a in zip(LEVELS, ability)}
    bank["method"] = ("Rasch-style model on all questions + each question's own pilot answers and faculty "
                      f"estimate (faculty = {TEACHER_WEIGHT} people, model = {MODEL_WEIGHT} people). "
                      "Levels: IR/PS students = technical; others >= 50% correct = medium, else easy.")
    with open(bank_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    print("Pilot participants by level:", people)
    print("Level abilities (easy fixed at 0):", bank["level_ability"])
    print(f"Updated p values for {len(items)} questions in {bank_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])