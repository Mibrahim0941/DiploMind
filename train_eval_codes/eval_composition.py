"""
DiploMind: composed adapters vs. jointly trained adapters.

Compares, for each of the 9 (language, depth) groups, on held-out test rows:

    composed   W = W0 + lambda_L * dW_lang(l) + lambda_D * dW_depth(d)      (6 adapters -> 9 behaviours)
    joint      one adapter trained only on that (language, depth) group     (9 adapters, the baseline)

plus reference points: base model, language-only, depth-only, and composed with lambda = (1, 1).
lambda_L, lambda_D come from a grid search on the VALIDATION rows (not the test rows).

Metric: teacher-forced loss on the reference answer (token-weighted mean negative log-likelihood over the
assistant tokens, i.e. perplexity = exp(loss)). Lower is better.

----------------------------------------------------------------------------------------------------
Before running
  1. The 6 component adapters exist in  adapters\\  (trained with TAG_SUFFIX = llama31_8b_4bit_v2):
        lang_eng, lang_ur, lang_urdish, depth_easy, depth_med, depth_hard
  2. The 9 joint adapters exist in adapters\\ , trained on all_dataset_joint with the SAME settings
     (same MAX_LEN / epochs / learning rate). With the trainer script (DATA_ROOT env var):
        $env:DATA_ROOT = "all_dataset_joint"; $env:TAG_SUFFIX = "llama31_8b_4bit_joint"
        python train_all_adapters_4bit.py
  3. all_dataset_joint\\<lang> <depth>\\{train,val,test}.csv  (e.g. "eng easy", "ur med", "urdish hard").
     Folder names may use spaces, underscores or dashes; see LANG_ALIASES / DEPTH_ALIASES below.

Run (PowerShell), from the folder that contains adapters\\ and all_dataset_joint\\ :
    $env:HF_TOKEN = "hf_xxx"
    python eval_composition.py

It is resumable: every finished measurement is cached in composition_results\\cache.json, so after a crash
or restart just run the same command again.

Optional environment variables:
    GRID="0.25,0.5,0.75,1.0,1.25"   lambda values tried for BOTH lambda_L and lambda_D
    VAL_ROWS=40                     validation rows per group used for the grid search
    MAX_LEN=...                     default: the max_len stored in the component adapters
    COMPONENT_TAG, JOINT_TAG        adapter folder suffixes
    COMPONENT_DATA, JOINT_DATA      data folders (default training_sets, all_dataset_joint)
"""
import gc
import itertools
import json
import math
import os
import random
import re
import time

import pandas as pd
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_DIR = os.path.join(HERE, "adapters")
JOINT_DATA = os.path.join(HERE, os.environ.get("JOINT_DATA", "all_dataset_joint"))
COMPONENT_DATA = os.path.join(HERE, os.environ.get("COMPONENT_DATA", "training_sets"))
COMPONENT_TAG = os.environ.get("COMPONENT_TAG", "llama31_8b_4bit_v2")
JOINT_TAG = os.environ.get("JOINT_TAG", "llama31_8b_4bit_joint")
OUT_DIR = os.path.join(HERE, "composition_results")
GRID = [float(x) for x in os.environ.get("GRID", "0.25,0.5,0.75,1.0,1.25").split(",")]
VAL_ROWS = int(os.environ.get("VAL_ROWS", 40))
SEED = 42
SYSTEM = "You are DiploMind, an assistant that explains news and international affairs to Pakistani readers."

# canonical key -> accepted spellings in folder names (whole words, case-insensitive)
LANG_ALIASES = {"eng": {"eng", "english", "en"}, "ur": {"ur", "urdu"}, "urdish": {"urdish", "romanurdu"}}
DEPTH_ALIASES = {"easy": {"easy"}, "med": {"med", "medium", "mid"}, "hard": {"hard"}}

METHOD_ORDER = ["base", "lang_only", "depth_only", "composed_1_1", "composed_global", "composed_percell", "joint"]
METHOD_LABEL = {"base": "base model", "lang_only": "language adapter only", "depth_only": "depth adapter only",
                "composed_1_1": "composed (lambda 1, 1)", "composed_global": "composed (global lambda)",
                "composed_percell": "composed (per-group lambda)", "joint": "joint (9 adapters)"}


# --------------------------------------------------------------------------- #
# Folder / adapter discovery
# --------------------------------------------------------------------------- #
def parse_name(name):
    """'eng easy' / 'lang_urdish_llama31_...' -> (language key or None, depth key or None)."""
    lang = depth = None
    for tok in re.split(r"[^a-z0-9]+", name.lower()):
        for k, aliases in LANG_ALIASES.items():
            if tok in aliases:
                lang = k
        for k, aliases in DEPTH_ALIASES.items():
            if tok in aliases:
                depth = k
    return lang, depth


def find_adapters(tag):
    """{folder stem: path} for every adapters\\<stem>_<tag> folder that holds a saved adapter."""
    found = {}
    if not os.path.isdir(ADAPTER_DIR):
        return found
    for d in sorted(os.listdir(ADAPTER_DIR)):
        if d.endswith("_" + tag) and os.path.isfile(os.path.join(ADAPTER_DIR, d, "adapter_config.json")):
            found[d[: -(len(tag) + 1)]] = os.path.join(ADAPTER_DIR, d)
    return found


def discover():
    """-> cells {(lang, depth): data folder}, langs {key: path}, depths {key: path}, joint {(lang, depth): path}"""
    cells = {}
    for d in sorted(os.listdir(JOINT_DATA)):
        p = os.path.join(JOINT_DATA, d)
        if os.path.isdir(p) and all(os.path.isfile(os.path.join(p, f"{s}.csv")) for s in ("train", "val", "test")):
            lang, depth = parse_name(d)
            if lang is None or depth is None:
                print(f"  skipping data folder '{d}': cannot tell language/depth (edit LANG_ALIASES/DEPTH_ALIASES)")
            else:
                cells[(lang, depth)] = d
    langs, depths = {}, {}
    for stem, path in find_adapters(COMPONENT_TAG).items():
        l, dp = parse_name(stem)
        if stem.startswith("lang_") and l:
            langs[l] = path
        elif stem.startswith("depth_") and dp:
            depths[dp] = path
    joint = {}
    for stem, path in find_adapters(JOINT_TAG).items():
        l, dp = parse_name(stem)
        if l and dp:
            joint[(l, dp)] = path
    return cells, langs, depths, joint


def label(cell):
    return f"{cell[0]}_{cell[1]}"


# --------------------------------------------------------------------------- #
# Composition: several LoRA adapters active at once, each scaled by its lambda
# --------------------------------------------------------------------------- #
def lora_modules(model):
    return [m for m in model.modules() if isinstance(getattr(m, "scaling", None), dict) and hasattr(m, "lora_A")]


def activate(model, weights):
    """Make exactly the adapters in `weights` ({name: lambda}) active. For every LoRA layer the output becomes
        W0 x + sum_i lambda_i * (alpha_i / r_i) * B_i A_i x      i.e.  W = W0 + sum_i lambda_i * dW_i
    PEFT adds up the outputs of all active adapters; lambda is applied by rescaling each adapter's LoRA scaling."""
    names = list(weights)
    model.base_model.set_adapter(names)
    for m in lora_modules(model):
        if not hasattr(m, "_dm_base"):
            m._dm_base = {}
        for n, lam in weights.items():
            if n in m.scaling:
                m._dm_base.setdefault(n, m.scaling[n])      # the untouched alpha / r of this adapter
                m.scaling[n] = m._dm_base[n] * lam


def forget(model, name):
    for m in lora_modules(model):
        getattr(m, "_dm_base", {}).pop(name, None)


def load_adapters(base, named_paths):
    from peft import PeftModel
    items = list(named_paths.items())
    model = PeftModel.from_pretrained(base, items[0][1], adapter_name=items[0][0], is_trainable=False)
    for name, path in items[1:]:
        model.load_adapter(path, adapter_name=name, is_trainable=False)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
@torch.no_grad()
def nll_on(model, rows, device):
    """-> (sum of negative log-likelihood, number of assistant tokens) over the encoded rows."""
    total, ntok = 0.0, 0
    for r in rows:
        ids = torch.tensor([r["input_ids"]], device=device)
        labels = torch.tensor([r["labels"]], device=device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        lab = labels[0, 1:]
        mask = lab != -100
        n = int(mask.sum())
        if n == 0:
            continue
        total += F.cross_entropy(logits[0, :-1][mask].float(), lab[mask], reduction="sum").item()
        ntok += n
        del logits
    return total, ntok


class Cache:
    """Finished measurements, saved after every one so a crash or restart loses nothing."""

    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.isfile(path) else {}

    def has(self, key):
        return key in self.d

    def get(self, key, fn):
        if key not in self.d:
            self.d[key] = list(fn())
            tmp = self.path + ".tmp"
            json.dump(self.d, open(tmp, "w"))
            os.replace(tmp, self.path)
        return self.d[key]


# --------------------------------------------------------------------------- #
# The experiment (kept free of tokenizer / model-loading code so it can be tested on a tiny model)
# --------------------------------------------------------------------------- #
def evaluate_all(model, data, joint_paths, device, out_dir, grid=GRID):
    """
    model       PeftModel with adapters named 'lang_<key>' and 'depth_<key>' loaded
    data        {(lang, depth): {"val": [encoded rows], "test": [encoded rows]}}
    joint_paths {(lang, depth): path of that group's jointly trained adapter}
    """
    os.makedirs(out_dir, exist_ok=True)
    cache = Cache(os.path.join(out_dir, "cache.json"))
    cells = sorted(data)
    pairs = list(itertools.product(grid, grid))
    t0 = time.time()

    def comp(cell, lL, lD):
        return {f"lang_{cell[0]}": lL, f"depth_{cell[1]}": lD}

    # ---- phase 1: grid search of (lambda_L, lambda_D) on the validation rows
    print(f"\n[phase 1] grid search: {len(cells)} groups x {len(pairs)} lambda pairs on validation rows")
    todo = sum(1 for c in cells for lL, lD in pairs if not cache.has(f"val|{label(c)}|{lL}|{lD}"))
    done = 0
    for c in cells:
        for lL, lD in pairs:
            key = f"val|{label(c)}|{lL}|{lD}"
            if not cache.has(key):
                activate(model, comp(c, lL, lD))
                cache.get(key, lambda: nll_on(model, data[c]["val"], device))
                done += 1
                if done % 5 == 0 or done == todo:
                    el = time.time() - t0
                    print(f"  {done}/{todo} grid points | {el / 60:.0f} min elapsed | ~{el / done * (todo - done) / 60:.0f} min left",
                          flush=True)

    def vloss(c, lL, lD):
        nll, n = cache.d[f"val|{label(c)}|{lL}|{lD}"]
        return nll / n

    global_best = min(pairs, key=lambda p: sum(vloss(c, *p) for c in cells) / len(cells))
    percell_best = {c: min(pairs, key=lambda p: vloss(c, *p)) for c in cells}
    print(f"  best global lambda (L, D) = {global_best}")
    for c in cells:
        print(f"  best for {label(c):14} = {percell_best[c]}   val loss {vloss(c, *percell_best[c]):.4f}")

    grid_df = pd.DataFrame([{"group": label(c), "lambda_L": lL, "lambda_D": lD, "val_loss": vloss(c, lL, lD)}
                            for c in cells for lL, lD in pairs])
    grid_df.to_csv(os.path.join(out_dir, "grid_val_losses.csv"), index=False)

    # ---- phase 2: test rows
    print("\n[phase 2] test rows")
    records = []

    def measure(c, method, lL, lD, fn):
        key = f"test|{label(c)}|{method}|{lL}|{lD}"
        nll, n = cache.get(key, fn)
        records.append({"group": label(c), "language": c[0], "depth": c[1], "method": method,
                        "lambda_L": lL, "lambda_D": lD, "test_loss": nll / n, "perplexity": math.exp(nll / n),
                        "n_rows": len(data[c]["test"]), "n_tokens": n})

    for i, c in enumerate(cells, 1):
        test = data[c]["test"]
        print(f"  group {i}/{len(cells)}: {label(c)} ({len(test)} test rows) | {(time.time() - t0) / 60:.0f} min elapsed", flush=True)
        run = lambda: nll_on(model, test, device)

        def base_fn():
            with model.disable_adapter():
                return nll_on(model, test, device)

        measure(c, "base", "", "", base_fn)
        activate(model, {f"lang_{c[0]}": 1.0}); measure(c, "lang_only", 1.0, 0.0, run)
        activate(model, {f"depth_{c[1]}": 1.0}); measure(c, "depth_only", 0.0, 1.0, run)
        activate(model, comp(c, 1.0, 1.0)); measure(c, "composed_1_1", 1.0, 1.0, run)
        gl, gd = global_best
        activate(model, comp(c, gl, gd)); measure(c, "composed_global", gl, gd, run)
        pl, pd_ = percell_best[c]
        activate(model, comp(c, pl, pd_)); measure(c, "composed_percell", pl, pd_, run)

        jkey = f"test|{label(c)}|joint||"
        if not cache.has(jkey):
            model.load_adapter(joint_paths[c], adapter_name="joint", is_trainable=False)
            activate(model, {"joint": 1.0})
            cache.get(jkey, run)
            activate(model, comp(c, 1.0, 1.0))               # move away from 'joint' before deleting it
            model.delete_adapter("joint")
            forget(model, "joint")
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        nll, n = cache.d[jkey]
        records.append({"group": label(c), "language": c[0], "depth": c[1], "method": "joint", "lambda_L": "",
                        "lambda_D": "", "test_loss": nll / n, "perplexity": math.exp(nll / n),
                        "n_rows": len(test), "n_tokens": n})

    write_reports(pd.DataFrame(records), grid_df, cells, global_best, out_dir)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def write_reports(df, grid_df, cells, global_best, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df.to_csv(os.path.join(out_dir, "results_test_long.csv"), index=False)
    wide = df.pivot(index="group", columns="method", values="test_loss")[METHOD_ORDER]
    wide = wide.loc[[label(c) for c in cells]]
    wide.loc["MEAN (groups equal)"] = wide.mean()
    wide.to_csv(os.path.join(out_dir, "results_test_loss.csv"))
    ppl = wide.map(math.exp) if hasattr(wide, "map") else wide.applymap(math.exp)
    ppl.to_csv(os.path.join(out_dir, "results_test_perplexity.csv"))

    diff = (wide["composed_global"] - wide["joint"]).iloc[:-1]
    diff_pc = (wide["composed_percell"] - wide["joint"]).iloc[:-1]
    lines = ["Test loss per group (lower is better)", "", wide.to_string(float_format=lambda x: f"{x:.3f}"), "",
             f"global lambda (L, D) = {global_best}", "",
             "composed (global lambda) minus joint, per group (negative = composition better):",
             diff.to_string(float_format=lambda x: f"{x:+.3f}"),
             f"composition better in {(diff < 0).sum()} of {len(diff)} groups | mean difference {diff.mean():+.3f}", "",
             f"composed (per-group lambda) minus joint: better in {(diff_pc < 0).sum()} of {len(diff_pc)} groups | "
             f"mean difference {diff_pc.mean():+.3f}"]
    text = "\n".join(lines)
    open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8").write(text)
    print("\n" + "=" * 100 + "\n" + text)

    # bar chart: loss per group and method
    show = ["base", "lang_only", "depth_only", "composed_global", "composed_percell", "joint"]
    body = wide.iloc[:-1][show]
    ax = body.rename(columns=METHOD_LABEL).plot(kind="bar", figsize=(max(9, 1.3 * len(body)), 4.8), width=0.85)
    ax.set_ylabel("test loss (lower is better)"); ax.set_xlabel("")
    ax.set_title("Composed adapters vs jointly trained adapters")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.3)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison.png"), dpi=140); plt.close()

    # heat-map of the grid search (validation loss, averaged over the groups)
    g = grid_df.groupby(["lambda_L", "lambda_D"]).val_loss.mean().unstack()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(g.values, origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(g.columns)), [f"{x:g}" for x in g.columns])
    ax.set_yticks(range(len(g.index)), [f"{x:g}" for x in g.index])
    ax.set_xlabel("lambda_D (depth)"); ax.set_ylabel("lambda_L (language)")
    ax.set_title("Mean validation loss over groups")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax.text(j, i, f"{g.values[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lambda_grid.png"), dpi=140); plt.close()
    print(f"\nFiles written to {out_dir}")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def check_leakage(cells):
    """Warn if a test article of a group also appears in the training data of the adapters that are compared."""
    pools = {}
    for d in os.listdir(COMPONENT_DATA):
        f = os.path.join(COMPONENT_DATA, d, "train.csv")
        if os.path.isfile(f):
            pools.setdefault("component adapters", set()).update(pd.read_csv(f, usecols=["input"]).fillna("")["input"])
    bad = False
    for c, folder in sorted(cells.items()):
        test_inputs = set(pd.read_csv(os.path.join(JOINT_DATA, folder, "test.csv"), usecols=["input"]).fillna("")["input"])
        own = set(pd.read_csv(os.path.join(JOINT_DATA, folder, "train.csv"), usecols=["input"]).fillna("")["input"])
        n_comp = len(test_inputs & pools.get("component adapters", set()))
        n_own = len(test_inputs & own)
        if n_comp or n_own:
            bad = True
            print(f"  WARNING {label(c)}: {n_comp} test articles also in the component adapters' training data, "
                  f"{n_own} in this group's joint training data -> the comparison is not clean")
    print("  leakage check: " + ("PROBLEMS FOUND (see above)" if bad else "no test article appears in any training set (exact match)"))


def main():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise SystemExit('No Hugging Face token. In PowerShell run:  $env:HF_TOKEN = "hf_xxx"  and start again.')
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is False -> CPU-only PyTorch (see the install notes).")

    cells, langs, depths, joint = discover()
    problems = []
    if not cells:
        problems.append(f"no group folders found in {JOINT_DATA}")
    for (l, d) in cells:
        if l not in langs: problems.append(f"missing component adapter lang_{l}")
        if d not in depths: problems.append(f"missing component adapter depth_{d}")
        if (l, d) not in joint: problems.append(f"missing joint adapter for '{l} {d}'")
    print("groups     :", ", ".join(label(c) for c in sorted(cells)))
    print("lang adapt.:", sorted(langs), "| depth adapt.:", sorted(depths), "| joint adapt.:", len(joint))
    if problems:
        raise SystemExit("Cannot start:\n  " + "\n  ".join(sorted(set(problems))) +
                         f"\n(found in {ADAPTER_DIR}: component tag '{COMPONENT_TAG}', joint tag '{JOINT_TAG}')")

    # ---- settings must match training
    cfgs = {}
    for p in list(langs.values()) + list(depths.values()) + list(joint.values()):
        f = os.path.join(p, "diplomind_config.json")
        if os.path.isfile(f):
            cfgs[p] = json.load(open(f))
    max_lens = {c.get("max_len") for c in cfgs.values()}
    MAX_LEN = int(os.environ.get("MAX_LEN") or min(m for m in max_lens if m) if max_lens - {None} else 2048)
    if len(max_lens) > 1:
        print(f"  WARNING: adapters were trained with different max_len {sorted(m for m in max_lens if m)}; using {MAX_LEN}")
    print(f"MAX_LEN = {MAX_LEN}")

    print("\nchecking for train/test overlap ...")
    check_leakage(cells)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if tok.pad_token is None or tok.pad_token == tok.eos_token:
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    EOT = "<|eot_id|>" if "<|eot_id|>" in tok.get_vocab() else tok.eos_token

    def prompt_ids(instruction, news):     # identical to the training script
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{instruction}\n\nNews:\n{news}"}]
        return tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]

    def encode(row):                        # identical to the training script
        answer = tok(row["output"] + EOT, add_special_tokens=False)["input_ids"]
        news_ids = tok(row["input"], add_special_tokens=False)["input_ids"]
        budget = MAX_LEN - len(answer) - len(prompt_ids(row["instruction"], ""))
        if budget < len(news_ids):
            news_ids = news_ids[:max(budget, 0)]
        p = prompt_ids(row["instruction"], tok.decode(news_ids))
        return {"input_ids": (p + answer)[:MAX_LEN], "labels": ([-100] * len(p) + answer)[:MAX_LEN]}

    data = {}
    for c, folder in sorted(cells.items()):
        load = lambda s: pd.read_csv(os.path.join(JOINT_DATA, folder, f"{s}.csv")).fillna("").to_dict("records")
        val = load("val"); random.Random(SEED).shuffle(val)
        data[c] = {"val": [encode(r) for r in val[:VAL_ROWS]], "test": [encode(r) for r in load("test")]}
        print(f"  {label(c):14} val {len(data[c]['val'])} rows | test {len(data[c]['test'])} rows")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kw = dict(device_map={"": 0}, attn_implementation="sdpa", token=HF_TOKEN,
              quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    try:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, **kw)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, **kw)
    # same preparation as in training (casts the non-quantised layers to fp32), so the numbers match the training runs
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)

    named = {f"lang_{k}": v for k, v in langs.items() if any(k == l for l, _ in cells)}
    named.update({f"depth_{k}": v for k, v in depths.items() if any(k == d for _, d in cells)})
    model = load_adapters(base, named)
    print(f"loaded {len(named)} component adapters | GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    n_val = sum(len(data[c]["val"]) for c in data) * len(GRID) ** 2
    n_test = sum(len(data[c]["test"]) for c in data) * 6 + len(data) * 0
    print(f"about {n_val + n_test} forward passes (~1 s each) -> roughly {(n_val + n_test) / 3600:.1f} h")
    evaluate_all(model, data, {c: joint[c] for c in cells}, "cuda", OUT_DIR)


if __name__ == "__main__":
    main()
