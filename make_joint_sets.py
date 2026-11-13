"""
DiploMind: build the 9 JOINT training sets (one per language x depth group) for the composition comparison.

Input : the same grid_<machine>.csv files as make_training_sets.py (9 rows per article).
Output: all_dataset_joint/
          en_easy/    en_medium/    en_hard/
          ur_easy/    ur_medium/    ur_hard/
          urdish_easy/ urdish_medium/ urdish_hard/
        each with train.csv, val.csv, test.csv   (+ summary.csv in the main folder)

Every joint adapter is trained on ONE of these groups only. They are compared with the composition of the
6 component adapters (training_sets/lang_* and depth_*), so the article split MUST be identical:
if training_sets/split_articles.csv exists, the split is read from it (so a train/val/test article is the same
in all 15 sets, even if the grid files changed since). Otherwise the split is computed exactly like
make_training_sets.py (same seed, same article-level, source-language-stratified procedure).

Run:  python make_joint_sets.py
"""
import os, re, glob, random, collections
import pandas as pd

# ---------------- settings ----------------
GRID_FILES = glob.glob("dataset_out/**/grid_*.csv", recursive=True) or glob.glob("grid_*.csv")
OUT_DIR    = "all_dataset_joint"
SPLIT_FILE = os.path.join("training_sets", "split_articles.csv")     # reused when it exists
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
arts = df.drop_duplicates("article_id")[["article_id", "src_lang"]]

# ---------------- article-level split: reuse the one of the 6 component sets ----------------
if os.path.isfile(SPLIT_FILE):
    saved = pd.read_csv(SPLIT_FILE)
    split_of = dict(zip(saved["article_id"], saved["split"]))
    missing = set(arts["article_id"]) - set(split_of)
    if missing:
        raise SystemExit(f"{len(missing)} articles are not in {SPLIT_FILE} (e.g. {sorted(missing)[:5]}). "
                         "The grid files changed since training_sets was made. Re-run make_training_sets.py AND retrain "
                         "the component adapters, or delete the split file to compute a fresh split.")
    extra = set(split_of) - set(arts["article_id"])
    print(f"Split read from {SPLIT_FILE} (identical to the 6 component sets)"
          + (f"; {len(extra)} articles in that file are no longer kept and are ignored" if extra else ""))
else:
    print(f"{SPLIT_FILE} not found: computing the split (same procedure and seed as make_training_sets.py)")
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

# ---------------- write the 9 joint sets ----------------
os.makedirs(OUT_DIR, exist_ok=True)
summary = []
for lang in LANGS:
    for depth in DEPTHS:
        name = f"{lang}_{depth}"
        folder = os.path.join(OUT_DIR, name)
        os.makedirs(folder, exist_ok=True)
        part = df[(df["language"] == lang) & (df["depth"] == depth)]
        for split in SPLIT:
            s = part[part["split"] == split].sample(frac=1, random_state=SEED)[COLS]   # shuffle rows
            s.to_csv(os.path.join(folder, f"{split}.csv"), index=False, encoding="utf-8-sig")
            summary.append({"set": name, "split": split, "rows": len(s), "articles": s["article_id"].nunique(),
                            "src_ur": int((s["src_lang"] == "ur").sum()), "src_en": int((s["src_lang"] == "en").sum())})
summary = pd.DataFrame(summary)
summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

# ---------------- checks ----------------
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    overlap = set(df.loc[df.split == a, "article_id"]) & set(df.loc[df.split == b, "article_id"])
    assert not overlap, f"article leak between {a} and {b}"
for split in SPLIT:                      # every group holds the same articles, one row each
    per_group = summary[summary.split == split]
    assert per_group["rows"].nunique() == 1 and (per_group["rows"] == per_group["articles"]).all(), \
        f"groups differ in size for {split}: {per_group[['set', 'rows', 'articles']].to_dict('records')}"
comp_test = os.path.join("training_sets", "lang_en", "test.csv")
if os.path.isfile(comp_test):            # same test articles as the component sets?
    for split in SPLIT:
        a = set(pd.read_csv(os.path.join("training_sets", "lang_en", f"{split}.csv"))["article_id"])
        b = set(pd.read_csv(os.path.join(OUT_DIR, "en_easy", f"{split}.csv"))["article_id"])
        assert a == b, f"{split}: the joint sets and training_sets contain different articles"
    print("Verified: train/val/test articles are identical to training_sets.")
print(f"\nArticles kept: {len(arts)} | split:", dict(collections.Counter(split_of[a] for a in arts['article_id'])))
print("No article appears in more than one split.\n")
print(summary.to_string(index=False))
print(f"\nSaved to {OUT_DIR}/")
