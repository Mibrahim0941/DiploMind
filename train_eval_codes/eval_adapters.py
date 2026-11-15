"""
DiploMind: before/after check of the trained adapters on GENERATED text (not only loss).

Runs the base Llama and each adapter on the same 12 test articles that the zero-shot comparison used
(same seed, same selection), with the same checks, so the numbers can be put side by side:
language/script followed, similarity to the 70B reference (chrF), length, truncation, markdown, English reading grade.

Run it after train_all_adapters_4bit.py, in the same folder (needs training_sets/ and adapters/):
    python eval_adapters.py

Resumable: every finished generation is appended to eval_generation/generations.jsonl and skipped on re-runs.
Settings (environment variables): TAG_SUFFIX, N_ARTICLES, SEED, MAX_NEW_TOKENS, GEN_BATCH, MAX_NEWS_TOKENS.
    pip install sacrebleu
"""
import os, re, json, random, time, gc
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "training_sets")
ADAPTERS_DIR = os.path.join(HERE, "adapters")
OUT_DIR = os.path.join(HERE, "eval_generation")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
TAG_SUFFIX = os.environ.get("TAG_SUFFIX", "llama31_8b_4bit_v2")
N_ARTICLES = int(os.environ.get("N_ARTICLES", 12))      # keep 12 and SEED 7 to get the same articles as the zero-shot run
SEED = int(os.environ.get("SEED", 7))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 900))
GEN_BATCH = int(os.environ.get("GEN_BATCH", 2))          # lower to 1 if the GPU runs out of memory
MAX_NEWS_TOKENS = int(os.environ.get("MAX_NEWS_TOKENS", 3500))   # safety cap on the article length
SYSTEM = "You are DiploMind, an assistant that explains news and international affairs to Pakistani readers."
SETS = ["lang_en", "lang_ur", "lang_urdish", "depth_easy", "depth_medium", "depth_hard"]


# --------------------------------------------------------------------------- #
# Data: the same sample as the zero-shot notebook
# --------------------------------------------------------------------------- #
def read_csv(p):
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def load_test_rows(data_root):
    # {adapter_name: dataframe of that adapter's test rows}
    return {s: read_csv(os.path.join(data_root, s, "test.csv")) for s in SETS
            if os.path.exists(os.path.join(data_root, s, "test.csv"))}


def pick_sample_ids(test_by_adapter, n_articles=N_ARTICLES, seed=SEED):
    # identical to the zero-shot notebook: merge the six slices, keep complete articles, half Urdu-source / half English-source
    grid = pd.concat(test_by_adapter.values()).drop_duplicates(["article_id", "language", "depth"])
    per = grid.groupby("article_id").size()
    grid = grid[grid["article_id"].isin(per[per == 9].index)]
    arts = grid.drop_duplicates("article_id")[["article_id", "src_lang"]]
    rng = random.Random(seed)
    pools = {k: sorted(arts[arts["src_lang"] == k]["article_id"]) for k in ("ur", "en")}   # order matters: ur first
    for k in pools:
        rng.shuffle(pools[k])
    return sorted(pools["ur"][:(n_articles + 1) // 2] + pools["en"][:n_articles // 2])


# --------------------------------------------------------------------------- #
# Metrics (same rules as the zero-shot notebook)
# --------------------------------------------------------------------------- #
ARABIC = re.compile(r"[؀-ۿ]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
URDU_FUNC = {"hai", "hain", "ka", "ki", "ke", "mein", "main", "aur", "ko", "se", "ne", "par", "nahi", "nahin",
             "yeh", "ye", "woh", "wo", "tha", "thi", "thay", "gaya", "gayi", "kiya", "kar", "liye", "bhi", "jo"}
HINDI = re.compile(r"\b(sarkar|shanti|shant|vishay|adhikari|hinsak|samasya|upyog|rashtra|suraksha|yuddh|mantri|pradhan|dhanyavad)\b", re.I)
MARKDOWN = re.compile(r"\*\*|^\s*[\*\-] |^\s*\d+\. |^#+ ", re.M)


def urdu_share(t):
    letters = [c for c in t if c.isalpha()]
    return sum(bool(ARABIC.match(c)) for c in letters) / max(len(letters), 1)


def roman_urdu_rate(t):
    toks = re.findall(r"[a-zA-Z]+", t.lower())
    return sum(w in URDU_FUNC for w in toks) / max(len(toks), 1)


def syllables(w):
    w = w.lower()
    n = len(re.findall(r"[aeiouy]+", w)) - (1 if w.endswith("e") and not w.endswith("le") else 0)
    return max(n, 1)


def fk_grade(t):
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    words = re.findall(r"[A-Za-z]+", t)
    if not sents or not words:
        return None
    return round(0.39 * len(words) / len(sents) + 11.8 * sum(map(syllables, words)) / len(words) - 15.59, 1)


def lang_ok(lang, t):
    us, rr = urdu_share(t), roman_urdu_rate(t)
    if lang == "en":
        return us == 0 and rr < 0.06
    if lang == "urdish":
        return us < 0.02 and rr >= 0.12
    return us >= 0.8 and not DEVANAGARI.search(t)


def score(row, output, n_new):
    import sacrebleu
    return {"lang_ok": lang_ok(row["language"], output),
            "chrf": round(sacrebleu.sentence_chrf(output, [row["output"]]).score, 1) if output else 0.0,
            "words": len(output.split()), "ref_words": len(row["output"].split()),
            "truncated": n_new >= MAX_NEW_TOKENS,
            "markdown": bool(MARKDOWN.search(output)),
            "hindi_words": len(HINDI.findall(output)),
            "fk": fk_grade(output) if row["language"] == "en" else None}


# --------------------------------------------------------------------------- #
# Generation (resumable)
# --------------------------------------------------------------------------- #
def key_of(who, r):
    return (who, r["article_id"], r["language"], r["depth"])


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                done[(d["who"], d["article_id"], d["language"], d["depth"])] = d
    return done


def build_prompt(tok, instruction, news):
    ids = tok(news, add_special_tokens=False)["input_ids"]
    if len(ids) > MAX_NEWS_TOKENS:
        news = tok.decode(ids[:MAX_NEWS_TOKENS])
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{instruction}\n\nNews:\n{news}"}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def run_generation(rows, who, tok, gen_fn, done, path, clear_fn=lambda: None):
    # rows: list of dict rows (instruction, input, ...). gen_fn(list_of_prompts) -> list of (text, n_new_tokens)
    todo = [r for r in rows if key_of(who, r) not in done]
    print(f"  {who}: {len(rows) - len(todo)}/{len(rows)} already done")
    if not todo:
        return
    prompts = {key_of(who, r): build_prompt(tok, r["instruction"], r["input"]) for r in todo}
    todo.sort(key=lambda r: len(prompts[key_of(who, r)]))
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), GEN_BATCH):
            batch = todo[b:b + GEN_BATCH]
            texts = [prompts[key_of(who, r)] for r in batch]
            try:
                res = gen_fn(texts)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                print("   out of memory: retrying one prompt at a time")
                clear_fn()
                res = []
                for x in texts:
                    res += gen_fn([x])
            for r, (txt, n) in zip(batch, res):
                rec = {"who": who, "article_id": r["article_id"], "language": r["language"], "depth": r["depth"],
                       "output": txt, "n_new": n}
                done[key_of(who, r)] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"   {who}: {min(b + GEN_BATCH, len(todo))}/{len(todo)} | {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(test_by_adapter, sample_ids, done):
    rows = []
    for name, df in test_by_adapter.items():
        sub = df[df["article_id"].isin(sample_ids)]
        for r in sub.to_dict("records"):
            for model_name, who in (("base", "base"), ("adapter", name)):
                d = done.get((who, r["article_id"], r["language"], r["depth"]))
                if d is None:
                    continue
                rows.append({"adapter": name, "model": model_name, "src_lang": r["src_lang"], "language": r["language"],
                             "depth": r["depth"], "article_id": r["article_id"], "output": d["output"],
                             **score(r, d["output"], d["n_new"])})
    res = pd.DataFrame(rows)
    out = []
    for (name, model_name), g in res.groupby(["adapter", "model"]):
        for part, gg in (("all", g), ("ur-source", g[g["src_lang"] == "ur"]), ("en-source", g[g["src_lang"] == "en"])):
            valid_fk = gg[(gg["language"] == "en") & gg["lang_ok"]]["fk"].dropna()
            out.append({"adapter": name, "model": model_name, "articles": part, "n": len(gg),
                        "lang_ok_%": round(100 * gg["lang_ok"].mean(), 1), "chrf": round(gg["chrf"].mean(), 1),
                        "words": round(gg["words"].mean()), "ref_words": round(gg["ref_words"].mean()),
                        "truncated_%": round(100 * gg["truncated"].mean(), 1),
                        "markdown_%": round(100 * gg["markdown"].mean(), 1),
                        "fk_en": round(valid_fk.mean(), 1) if len(valid_fk) else None})
    return res, pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Main (needs the GPU)
# --------------------------------------------------------------------------- #
def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit('No Hugging Face token. In PowerShell run:  $env:HF_TOKEN = "hf_xxx"')
    os.makedirs(OUT_DIR, exist_ok=True)
    test_by_adapter = load_test_rows(DATA_ROOT)
    sample_ids = pick_sample_ids(test_by_adapter)
    pd.DataFrame({"article_id": sample_ids}).to_csv(os.path.join(OUT_DIR, "sample_articles.csv"), index=False)
    adapters = [a for a in test_by_adapter if os.path.isdir(os.path.join(ADAPTERS_DIR, f"{a}_{TAG_SUFFIX}"))]
    missing = [a for a in test_by_adapter if a not in adapters]
    print(f"{len(sample_ids)} articles | adapters found: {adapters}" + (f" | MISSING: {missing}" if missing else ""))

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    tok.padding_side = "left"
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    kw = dict(device_map={"": 0}, attn_implementation="sdpa", token=token,
              quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    try:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, **kw)          # transformers >= 4.56
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, **kw)
    model = None
    for a in adapters:
        path = os.path.join(ADAPTERS_DIR, f"{a}_{TAG_SUFFIX}")
        if model is None:
            model = PeftModel.from_pretrained(base, path, adapter_name=a)
        else:
            model.load_adapter(path, adapter_name=a)
    model.eval()

    @torch.no_grad()
    def gen_fn(prompts):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id,
                             eos_token_id=eot_id, use_cache=True)
        new = out[:, enc["input_ids"].shape[1]:]
        return [(tok.decode(n, skip_special_tokens=True).strip(), int((n != tok.pad_token_id).sum())) for n in new]

    def clear():
        gc.collect(); torch.cuda.empty_cache()

    path = os.path.join(OUT_DIR, "generations.jsonl")
    done = load_done(path)
    # base model: each distinct (article, language, depth) once, even though two adapters share some of them
    sample_rows = {}
    for df in test_by_adapter.values():
        for r in df[df["article_id"].isin(sample_ids)].to_dict("records"):
            sample_rows[(r["article_id"], r["language"], r["depth"])] = r
    print(f"\nBase model on {len(sample_rows)} prompts")
    with model.disable_adapter():
        run_generation(list(sample_rows.values()), "base", tok, gen_fn, done, path, clear)
    for a in adapters:
        model.set_adapter(a)
        rows = test_by_adapter[a][test_by_adapter[a]["article_id"].isin(sample_ids)].to_dict("records")
        print(f"\nAdapter {a} on {len(rows)} prompts")
        run_generation(rows, a, tok, gen_fn, done, path, clear)

    res, summary = build_report(test_by_adapter, sample_ids, done)
    res.to_csv(os.path.join(OUT_DIR, "eval_results.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(OUT_DIR, "eval_summary.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n" + summary.to_string(index=False))
    print(f"\nSaved to {OUT_DIR}: eval_summary.csv (the before/after table), eval_results.csv (every output), generations.jsonl")


if __name__ == "__main__":
    main()
