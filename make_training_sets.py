"""
DiploMind: turn the generated grid into 6 training sets.

Input : one or more grid_<machine>.csv files (9 rows per article: 3 languages x 3 depths).
Output: training_sets/
          lang_en/      all English outputs, every depth
          lang_ur/      all Urdu outputs, every depth
          lang_urdish/  all Urdish outputs, every depth
          depth_easy/   all easy outputs, every language
          depth_medium/ all medium outputs, every language
          depth_hard/   all hard outputs, every language
        each with train.csv, val.csv, test.csv
        plus split_articles.csv (which article went to which split) and summary.csv

The split is done by ARTICLE, not by row, and it is the same for all 6 sets:
an article in test is in test everywhere, so no input is ever seen in training
and then tested on, and the adapters can be compared on exactly the same test articles.

Works on Kaggle (one cell) or locally:  python make_training_sets.py
"""
import os, re, glob, random, collections
import pandas as pd

# ---------------- settings ----------------
GRID_FILES = glob.glob("dataset_out/**/grid_*.csv", recursive=True) or glob.glob("grid_*.csv")
OUT_DIR    = "training_sets"
SPLIT      = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED       = 42

LANGS  = ["en", "ur", "urdish"]
DEPTHS = ["easy", "medium", "hard"]

LANG_NAME = {
    "en":     "English",
    "ur":     "Urdu (Urdu script)",
    "urdish": "Urdish (Urdu in Roman letters, mixed with common English words)",
}
DEPTH_READER = {
    "easy":   "a reader with no knowledge of international relations: explain the key terms and actors in simple words",
    "medium": "a reader with some knowledge of international relations: briefly explain only the most important concepts",
    "hard":   "a reader familiar with international relations: use the technical terms without explaining them",
}

def instruction(lang, depth):
    return f"Explain this news in {LANG_NAME[lang]} for {DEPTH_READER[depth]}."

# ---------------- load ----------------
assert GRID_FILES, "No grid_*.csv found: set GRID_FILES to your file paths."
df = pd.concat([pd.read_csv(f) for f in GRID_FILES], ignore_index=True)
print(f"Loaded {len(df)} rows from {len(GRID_FILES)} file(s): {GRID_FILES}")
df = df.drop_duplicates(subset=["article_id", "language", "depth"])
df["output"] = df["output"].fillna("").astype(str).str.strip()
df["input"] = df["input"].fillna("").astype(str).str.strip()

# ---------------- quality filter (whole article goes if any of its 9 cells is bad) ----------------
END = re.compile(r"""[.!?۔؟"”'’)]$""")
def bad_reason(row):
    t = row["output"]
    if not t:
        return "empty"
    if not END.search(t):
        return "cut_off"
    if "<think>" in t or "</think>" in t:
        return "think_tag"
    if re.search(r"[ऀ-ॿ]", t):
        return "devanagari"
    if row["language"] != "ur" and re.search(r"[؀-ۿ]{3,}", t):
        return "urdu_script_in_latin_output"
    if len(t.split()) < 40:
        return "too_short"
    return None

df["bad"] = df.apply(bad_reason, axis=1)
complete = df.groupby("article_id").size() == len(LANGS) * len(DEPTHS)
bad_articles = set(df.loc[df["bad"].notna(), "article_id"]) | set(complete[~complete].index)
print("Bad cells:", dict(collections.Counter(df["bad"].dropna())))
print(f"Articles removed: {len(bad_articles)} (any bad or missing cell)")
df = df[~df["article_id"].isin(bad_articles)].copy()

# ---------------- article-level split, stratified by source language ----------------
arts = df.drop_duplicates("article_id")[["article_id", "src_lang"]]
rng = random.Random(SEED)
split_of = {}
for _, group in arts.groupby("src_lang"):
    ids = sorted(group["article_id"])
    rng.shuffle(ids)
    n = len(ids)
    n_train, n_val = round(n * SPLIT["train"]), round(n * SPLIT["val"])
    for i, a in enumerate(ids):
        split_of[a] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
df["split"] = df["article_id"].map(split_of)

df["instruction"] = [instruction(l, d) for l, d in zip(df["language"], df["depth"])]
COLS = ["instruction", "input", "output", "language", "depth", "src_lang", "article_id", "url"]

# ---------------- write the 6 sets ----------------
os.makedirs(OUT_DIR, exist_ok=True)
sets = {f"lang_{l}": df[df["language"] == l] for l in LANGS}
sets.update({f"depth_{d}": df[df["depth"] == d] for d in DEPTHS})

summary = []
for name, part in sets.items():
    folder = os.path.join(OUT_DIR, name)
    os.makedirs(folder, exist_ok=True)
    for split in SPLIT:
        s = part[part["split"] == split].sample(frac=1, random_state=SEED)[COLS]   # shuffle rows
        s.to_csv(os.path.join(folder, f"{split}.csv"), index=False, encoding="utf-8-sig")
        summary.append({"set": name, "split": split, "rows": len(s),
                        "articles": s["article_id"].nunique(),
                        "src_ur": int((s["src_lang"] == "ur").sum()), "src_en": int((s["src_lang"] == "en").sum()),
                        **{f"lang_{l}": int((s["language"] == l).sum()) for l in LANGS},
                        **{f"depth_{d}": int((s["depth"] == d).sum()) for d in DEPTHS}})

df.drop_duplicates("article_id")[["article_id", "url", "src_lang", "split"]].to_csv(
    os.path.join(OUT_DIR, "split_articles.csv"), index=False, encoding="utf-8-sig")
summary = pd.DataFrame(summary)
summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

# ---------------- checks ----------------
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    overlap = set(df.loc[df.split == a, "article_id"]) & set(df.loc[df.split == b, "article_id"])
    assert not overlap, f"article leak between {a} and {b}"
print(f"\nArticles kept: {len(arts)} | split:", dict(collections.Counter(split_of.values())))
print("No article appears in more than one split.\n")
print(summary.to_string(index=False))
print(f"\nSaved to {OUT_DIR}/")
