"""
DiploMind: train ALL adapters one after another on Llama 3.1 8B Instruct (4-bit QLoRA), plain transformers + peft.
Version 2: longer context, more epochs with early stopping, best checkpoint kept, test loss on ALL test rows.

Put this file next to the `training_sets` folder and run:
    python train_all_adapters_4bit.py

Every sub-folder of training_sets that contains train.csv / val.csv / test.csv is trained
(e.g. lang_urdish, depth_easy, ...). Finished adapters are skipped; interrupted ones resume
from their last checkpoint, so you can just run the same command again.

Joint adapters (one per language/depth group, for the composition comparison):
    $env:DATA_ROOT = "all_dataset_joint"; $env:TAG_SUFFIX = "llama31_8b_4bit_joint"; python train_all_adapters_4bit.py

Settings can be changed without editing the file, with environment variables (PowerShell example):
    $env:EPOCHS = "3"; $env:MAX_LEN = "4096"; python train_all_adapters_4bit.py

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install transformers peft datasets accelerate bitsandbytes pandas matplotlib

The model is gated: request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct first,
then create a token (read access) at https://huggingface.co/settings/tokens.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # ignored on Windows, harmless

import gc, glob, inspect, json, math, random, shutil, time, traceback
import torch, pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments)

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, os.environ.get("DATA_ROOT", "training_sets"))   # e.g. $env:DATA_ROOT = "all_dataset_joint"
OUT_ROOT = HERE

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"     # official Meta repo (gated: your HF account needs approved access)

# Hugging Face token: set the HF_TOKEN environment variable (do NOT paste it into this file).
#   PowerShell:  $env:HF_TOKEN = "hf_xxx"
HF_TOKEN = os.environ.get("HF_TOKEN")

# New tag, so the first-round adapters (llama31_8b_4bit) are neither skipped nor overwritten.
TAG_SUFFIX = os.environ.get("TAG_SUFFIX", "llama31_8b_4bit_v2")
EPOCHS = int(os.environ.get("EPOCHS", 2))              # round 1 used 1; validation loss was still falling
PATIENCE = int(os.environ.get("PATIENCE", 3))          # stop early after this many evaluations without improvement
LR = 2e-4
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
BATCH, GRAD_ACCUM = 1, 8      # effective batch = 8
EVAL_ROWS = 60                # validation rows used during training (keeps each evaluation short)
EVAL_EVERY = SAVE_EVERY = 25   # must be equal: the best checkpoint is chosen from the evaluations
SEED = 42
SYSTEM = "You are DiploMind, an assistant that explains news and international affairs to Pakistani readers."

if not HF_TOKEN:
    raise SystemExit('No Hugging Face token. In PowerShell run:  $env:HF_TOKEN = "hf_xxx"  and start the script again.')
os.environ["HF_TOKEN"] = HF_TOKEN   # so every hub call (incl. peft's checkpoint save) is authenticated
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False -> CPU-only PyTorch. Install a CUDA build (see header).")
assert EVAL_EVERY == SAVE_EVERY, "EVAL_EVERY and SAVE_EVERY must be equal"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {torch.cuda.get_device_name(0)} ({GPU_GB:.1f} GB)")


def pick_max_len():
    # tokens per example. Round 1 used 2048 and cut the article in about a quarter of the rows (about half of the
    # Urdu-script articles). Longer is better, but 3072 overflowed a 16 GB card, so it follows the GPU memory.
    if os.environ.get("MAX_LEN"):
        return int(os.environ["MAX_LEN"])
    return 4096 if GPU_GB >= 40 else 3072 if GPU_GB >= 22 else 2048


MAX_LEN = pick_max_len()
print(f"MAX_LEN = {MAX_LEN} | epochs <= {EPOCHS} (early stopping patience {PATIENCE}) | tag {TAG_SUFFIX}")

adapters = sorted(d for d in os.listdir(DATA_ROOT)
                  if all(os.path.isfile(os.path.join(DATA_ROOT, d, f"{s}.csv")) for s in ("train", "val", "test")))
if not adapters:
    raise SystemExit(f"No adapter folders (with train/val/test.csv) found in {DATA_ROOT}")
print("Adapters to train:", adapters)

tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tok.pad_token is None or tok.pad_token == tok.eos_token:
    tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
tok.padding_side = "right"
EOT = "<|eot_id|>" if "<|eot_id|>" in tok.get_vocab() else tok.eos_token


# --------------------------------------------------------------------------- #
# Data helpers (loss on the assistant turn only; news is cut to fit MAX_LEN, the answer never)
# --------------------------------------------------------------------------- #
def prompt_ids(instruction, news):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{instruction}\n\nNews:\n{news}"}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def encode(row):
    answer = tok(row["output"] + EOT, add_special_tokens=False)["input_ids"]
    news_ids = tok(row["input"], add_special_tokens=False)["input_ids"]
    budget = MAX_LEN - len(answer) - len(prompt_ids(row["instruction"], ""))
    cut = budget < len(news_ids)
    if cut:
        news_ids = news_ids[:max(budget, 0)]
    p = prompt_ids(row["instruction"], tok.decode(news_ids))
    return {"input_ids": (p + answer)[:MAX_LEN], "labels": ([-100] * len(p) + answer)[:MAX_LEN], "cut": cut}


class FreeMemory(TrainerCallback):
    """Release cached GPU memory after each evaluation, just before the checkpoint is saved."""
    def on_evaluate(self, args, state, control, **kw):
        gc.collect(); torch.cuda.empty_cache()


def load_split(adapter, split):
    return pd.read_csv(os.path.join(DATA_ROOT, adapter, f"{split}.csv")).fillna("").to_dict("records")


def load_model():
    kw = dict(device_map={"": 0}, attn_implementation="sdpa", token=HF_TOKEN,
              quantization_config=BitsAndBytesConfig(
                  load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                  bnb_4bit_compute_dtype=DTYPE))
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, **kw)         # transformers >= 4.56
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, **kw)   # older versions
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    return get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))


# --------------------------------------------------------------------------- #
# One full run for one adapter
# --------------------------------------------------------------------------- #
def train_adapter(adapter):
    out_dir = os.path.join(OUT_ROOT, "adapters", f"{adapter}_{TAG_SUFFIX}")
    ckpt_dir = os.path.join(OUT_ROOT, "checkpoints", f"{adapter}_{TAG_SUFFIX}")
    cfg_path = os.path.join(out_dir, "diplomind_config.json")
    if os.path.isfile(cfg_path) and json.load(open(cfg_path)).get("finished"):
        print(f"\n### {adapter}: already finished, skipping")
        return
    os.makedirs(out_dir, exist_ok=True)
    random.seed(SEED); torch.manual_seed(SEED)
    print(f"\n{'#' * 100}\n### Training adapter: {adapter}\n{'#' * 100}")

    model = load_model()
    model.print_trainable_parameters()

    data = {s: [encode(r) for r in load_split(adapter, s)] for s in ("train", "val", "test")}
    for s, rows in data.items():
        lens = sorted(len(r["input_ids"]) for r in rows)
        n_cut = sum(r["cut"] for r in rows)
        print(f"{s:5}: {len(rows):5} rows | tokens median {lens[len(lens)//2]}, p95 {lens[int(len(lens)*.95)]}, "
              f"max {lens[-1]} | news cut: {n_cut} ({100 * n_cut / len(rows):.1f}%)")
    cut_train_pct = 100 * sum(r["cut"] for r in data["train"]) / len(data["train"])
    if cut_train_pct > 5:
        print(f"WARNING: the article is cut in {cut_train_pct:.0f}% of the training rows, so the reference answer "
              f"mentions facts the model cannot see. Raise MAX_LEN if the GPU memory allows (now {MAX_LEN}).")

    cols = ["input_ids", "labels"]
    mk = lambda rows: Dataset.from_list([{k: r[k] for k in cols} for r in rows])
    train_ds = mk(data["train"])
    val_rows = data["val"][:]; random.Random(SEED).shuffle(val_rows)
    val_ds = mk(val_rows[:EVAL_ROWS])
    test_ds = mk(data["test"])                    # ALL test rows (round 1 used only 60)
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)

    steps_total = math.ceil(len(train_ds) / (BATCH * GRAD_ACCUM)) * EPOCHS
    kw = dict(
        output_dir=ckpt_dir, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * steps_total)), weight_decay=0.0, optim="adamw_torch",
        bf16=DTYPE == torch.bfloat16, fp16=DTYPE == torch.float16, max_grad_norm=0.3, logging_steps=5,
        eval_strategy="steps", eval_steps=EVAL_EVERY,
        save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", seed=SEED,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0 if os.name == "nt" else 2,
    )
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" not in valid and "evaluation_strategy" in valid:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    targs = TrainingArguments(**{k: v for k, v in kw.items() if k in valid})
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator,
                      callbacks=[FreeMemory(), EarlyStoppingCallback(early_stopping_patience=PATIENCE)])

    last = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
    print(f"up to {steps_total} optimizer steps" + (f" | resuming from {last[-1]}" if last else ""))
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last[-1] if last else None)
    # train() returns normally when all steps are done OR early stopping fired; an interruption raises instead
    finished = trainer.state.global_step >= trainer.state.max_steps or trainer.control.should_training_stop
    print(f"Training time: {(time.time() - t0) / 3600:.2f} h | stopped at step {trainer.state.global_step}"
          f"/{trainer.state.max_steps} | best {trainer.state.best_model_checkpoint} "
          f"(val loss {trainer.state.best_metric}) | peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # ---- save adapter (the best checkpoint is loaded back into the model by load_best_model_at_end) + loss curve
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    hist = pd.DataFrame(trainer.state.log_history)
    hist.to_csv(os.path.join(out_dir, "log_history.csv"), index=False)
    tr = hist.dropna(subset=["loss"])[["step", "loss"]] if "loss" in hist else pd.DataFrame(columns=["step", "loss"])
    ev = hist.dropna(subset=["eval_loss"])[["step", "eval_loss"]] if "eval_loss" in hist else pd.DataFrame(columns=["step", "eval_loss"])
    print(ev.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(tr.step, tr.loss, label="train loss")
    plt.plot(ev.step, ev.eval_loss, "o-", label="validation loss")
    plt.xlabel("step"); plt.ylabel("loss"); plt.title(adapter); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=120, bbox_inches="tight"); plt.close()

    # ---- test loss on ALL test rows: base vs adapter
    with_adapter = trainer.evaluate(test_ds, metric_key_prefix="test")["test_loss"]
    with model.disable_adapter():
        base = trainer.evaluate(test_ds, metric_key_prefix="test")["test_loss"]
    print(f"Test loss  base: {base:.3f} (ppl {math.exp(base):.1f}) | {adapter}: {with_adapter:.3f} (ppl {math.exp(with_adapter):.1f})")
    json.dump({"test_loss_base": base, "test_loss_adapter": with_adapter, "test_rows": len(test_ds)},
              open(os.path.join(out_dir, "test_loss.json"), "w"), indent=1)

    # ---- generate: base vs adapter on 3 test articles (a quick look; eval_adapters.py does the full check)
    model.eval()
    model.config.use_cache = True
    raw_test = load_split(adapter, "test"); random.Random(1).shuffle(raw_test)
    news_cap = max(MAX_LEN - 1100, 1200)

    def generate(instruction, news, max_new_tokens=1024):
        ids = torch.tensor([prompt_ids(instruction, news)], device=model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                 max_length=None, do_sample=False, repetition_penalty=1.05, use_cache=True,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids(EOT))
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    samples = []
    for r in raw_test[:3]:
        news = tok.decode(tok(r["input"], add_special_tokens=False)["input_ids"][:news_cap])
        tuned = generate(r["instruction"], news)
        with model.disable_adapter():
            base_out = generate(r["instruction"], news)
        samples.append({"instruction": r["instruction"], "language": r["language"], "depth": r["depth"],
                        "url": r["url"], "reference": r["output"], "base": base_out, "adapter": tuned})
        print("=" * 100, f"\n[{r['language']} / {r['depth']}] {r['url']}")
        print("\n--- BASE ---\n", base_out[:800])
        print("\n--- ADAPTER ---\n", tuned[:800])
    pd.DataFrame(samples).to_csv(os.path.join(out_dir, "samples.csv"), index=False, encoding="utf-8-sig")

    # ---- config last: its "finished" flag is what makes re-runs skip this adapter
    json.dump({"adapter": adapter, "base_model": MODEL_ID, "precision": "4bit-nf4", "system": SYSTEM,
               "max_len": MAX_LEN, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "epochs_planned": EPOCHS, "lr": LR,
               "steps": trainer.state.global_step, "max_steps": trainer.state.max_steps,
               "best_checkpoint": trainer.state.best_model_checkpoint, "best_val_loss": trainer.state.best_metric,
               "finished": finished, "train_rows": len(data["train"]),
               "train_rows_news_cut": sum(r["cut"] for r in data["train"]),
               "trained_with": "transformers+peft"}, open(cfg_path, "w"), indent=1)
    zip_path = shutil.make_archive(os.path.join(OUT_ROOT, f"adapter_{adapter}_{TAG_SUFFIX}"), "zip", out_dir)
    if finished:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    print("Packaged:", zip_path, f"({os.path.getsize(zip_path) / 1e6:.0f} MB) | finished: {finished}")

    del trainer, model
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# Run everything
# --------------------------------------------------------------------------- #
results = {}
for name in adapters:
    try:
        train_adapter(name)
        results[name] = "ok"
    except Exception:
        traceback.print_exc()
        results[name] = "FAILED"
    gc.collect(); torch.cuda.empty_cache()   # after the except block, so the failed run's model is released

print("\n" + "=" * 100 + "\nSummary:")
for name, status in results.items():
    print(f"  {name}: {status}")
