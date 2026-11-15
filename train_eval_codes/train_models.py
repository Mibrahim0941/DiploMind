"""
DiploMind: fast QLoRA adapters on Llama 3.1 8B Instruct (Unsloth)

Trains one LoRA adapter per run on a 4-bit Llama 3.1 8B Instruct.

    python train_models.py --adapter lang_urdish
    python train_models.py --adapter depth_easy

Requirements (install once, NOT inside this script):
    pip install unsloth
    # Local machine: make sure torch is a CUDA build, e.g.
    pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

Data layout: <data-root>/<adapter>/{train,val,test}.csv
(columns: instruction, input, output, language, depth, url)
"""
import os

# Must be set before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Unsloth trains on one GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # ignored on Windows, harmless

import argparse
import glob
import json
import math
import random
import shutil
import time

import torch
import pandas as pd

import unsloth  # noqa: F401  (must be imported before transformers)
from unsloth import FastLanguageModel, is_bfloat16_supported

# --------------------------------------------------------------------------- #
# 2. Settings
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--adapter", default="lang_urdish",
                    help='"lang_urdish" or "depth_easy" (any folder name in training_sets works)')
parser.add_argument("--data-root", default=None, help="folder that contains the <adapter>/ folders")
parser.add_argument("--out-root", default=None)
args_cli, _ = parser.parse_known_args()

ADAPTER = args_cli.adapter
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# Llama 3.2 3B baseline (about 2.5x faster): "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

DATA_ROOT = args_cli.data_root or (
    glob.glob("/kaggle/input/**/training_sets", recursive=True) or ["training_sets"])[0]
OUT_ROOT = args_cli.out_root or ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")

MAX_LEN = 3072        # tokens per example
EPOCHS = 2
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
BATCH = 1
GRAD_ACCUM = 8        # effective batch = 8
EVAL_ROWS = 60        # validation subset used during training
EVAL_EVERY = 50       # steps
SAVE_EVERY = 25       # steps (checkpoint for resuming)
TIME_LIMIT_H = 10.5   # stop and save before Kaggle's 12 h limit
SEED = 42

SYSTEM = "You are DiploMind, an assistant that explains news and international affairs to Pakistani readers."

TAG = ADAPTER + ("" if "8B" in MODEL_ID else "_" + MODEL_ID.split("/")[-1].split("-Instruct")[0].lower())
OUT_DIR = os.path.join(OUT_ROOT, "adapters", TAG)
CKPT_DIR = os.path.join(OUT_ROOT, "checkpoints", TAG)
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False - you have a CPU-only PyTorch build. "
                     "Reinstall a CUDA build of torch (see header of this file).")

print("Model:", MODEL_ID, "| adapter:", TAG)
print("Data :", os.path.join(DATA_ROOT, ADAPTER))
print("GPU  :", torch.cuda.get_device_name(0), "| bf16:", is_bfloat16_supported())

# --------------------------------------------------------------------------- #
# 3. Load the model (4-bit) and add LoRA
# --------------------------------------------------------------------------- #
model, tok = FastLanguageModel.from_pretrained(MODEL_ID, max_seq_length=MAX_LEN, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED,
)
if tok.pad_token is None or tok.pad_token == tok.eos_token:
    tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
tok.padding_side = "right"
model.print_trainable_parameters()
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# --------------------------------------------------------------------------- #
# 4. Load the data
# Each row becomes a chat: system, user (instruction + news), assistant (target).
# Loss is computed only on the assistant turn.
# --------------------------------------------------------------------------- #
def load(split):
    return pd.read_csv(os.path.join(DATA_ROOT, ADAPTER, f"{split}.csv")).fillna("").to_dict("records")


def user_text(instruction, news):
    return f"{instruction}\n\nNews:\n{news}"


def prompt_ids(instruction, news):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(instruction, news)}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


EOT = "<|eot_id|>" if "<|eot_id|>" in tok.get_vocab() else tok.eos_token


def encode(row):
    """Prompt + answer ids; labels masked on the prompt. News is cut to fit MAX_LEN, the answer never."""
    answer = tok(row["output"] + EOT, add_special_tokens=False)["input_ids"]
    news_ids = tok(row["input"], add_special_tokens=False)["input_ids"]
    budget = MAX_LEN - len(answer) - len(prompt_ids(row["instruction"], ""))
    cut = budget < len(news_ids)
    if cut:
        news_ids = news_ids[:max(budget, 0)]
    p = prompt_ids(row["instruction"], tok.decode(news_ids))
    return {"input_ids": (p + answer)[:MAX_LEN], "labels": ([-100] * len(p) + answer)[:MAX_LEN], "cut": cut}


data = {s: [encode(r) for r in load(s)] for s in ("train", "val", "test")}
for s, rows in data.items():
    lens = sorted(len(r["input_ids"]) for r in rows)
    print(f"{s:5}: {len(rows):5} rows | tokens median {lens[len(lens)//2]}, p95 {lens[int(len(lens)*.95)]}, "
          f"max {lens[-1]} | news cut: {sum(r['cut'] for r in rows)} | total {sum(lens)/1e6:.2f}M tokens")

# --------------------------------------------------------------------------- #
# 5. Train (speed report after 10 steps, time guard, resumable)
# --------------------------------------------------------------------------- #
import inspect
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from datasets import Dataset

cols = ["input_ids", "labels"]
train_ds = Dataset.from_list([{k: r[k] for k in cols} for r in data["train"]])
val_rows = data["val"][:]
random.Random(SEED).shuffle(val_rows)
val_ds = Dataset.from_list([{k: r[k] for k in cols} for r in val_rows[:EVAL_ROWS]])
collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)


class SpeedAndTimeGuard(TrainerCallback):
    """Prints measured speed and projected time after 10 steps; stops and saves before the time limit."""

    def on_train_begin(self, args, state, control, **kw):
        self.t0, self.s0 = time.time(), state.global_step

    def on_step_end(self, args, state, control, **kw):
        done = state.global_step - self.s0
        elapsed = time.time() - self.t0
        if done == 10:
            per_step = elapsed / done
            left = (state.max_steps - state.global_step) * per_step
            print(f"\n[speed] {per_step:.0f} s per step -> about {left / 3600:.1f} h left for {state.max_steps} steps")
        if elapsed > TIME_LIMIT_H * 3600:
            print(f"\n[time guard] {TIME_LIMIT_H} h reached at step {state.global_step}: saving and stopping. "
                  "Re-run the script to continue from this checkpoint.")
            control.should_save = True
            control.should_training_stop = True
        return control


steps_total = math.ceil(len(train_ds) / (BATCH * GRAD_ACCUM)) * EPOCHS
kw = dict(
    output_dir=CKPT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=max(1, int(0.05 * steps_total)),
    weight_decay=0.0,
    optim="adamw_8bit",
    bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
    max_grad_norm=0.3,
    logging_steps=5,
    eval_strategy="steps", eval_steps=EVAL_EVERY,
    save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=2,
    report_to="none",
    seed=SEED,
    # Windows can't fork DataLoader workers from a plain script -> use 0 there.
    dataloader_num_workers=0 if os.name == "nt" else 2,
)
valid = set(inspect.signature(TrainingArguments.__init__).parameters)
if "eval_strategy" not in valid and "evaluation_strategy" in valid:
    kw["evaluation_strategy"] = kw.pop("eval_strategy")
skipped = [k for k in kw if k not in valid]
if skipped:
    print("Skipped (not supported by this transformers version):", skipped)
targs = TrainingArguments(**{k: v for k, v in kw.items() if k in valid})

trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=collator, callbacks=[SpeedAndTimeGuard()])

last = sorted(glob.glob(os.path.join(CKPT_DIR, "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
print(f"{steps_total} optimizer steps | resuming from {last[-1]}" if last else f"{steps_total} optimizer steps")
t0 = time.time()
result = trainer.train(resume_from_checkpoint=last[-1] if last else None)
FINISHED = trainer.state.global_step >= trainer.state.max_steps
print(f"Training time this session: {(time.time() - t0) / 3600:.2f} h | finished: {FINISHED}")

# --------------------------------------------------------------------------- #
# 6. Save the adapter and loss curve
# --------------------------------------------------------------------------- #
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
with open(os.path.join(OUT_DIR, "diplomind_config.json"), "w") as f:
    json.dump({"adapter": ADAPTER, "base_model": MODEL_ID, "system": SYSTEM, "max_len": MAX_LEN,
               "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "epochs": EPOCHS, "lr": LR,
               "steps": trainer.state.global_step, "finished": FINISHED,
               "train_rows": len(data["train"]), "trained_with": "unsloth"}, f, indent=1)

hist = pd.DataFrame(trainer.state.log_history)
hist.to_csv(os.path.join(OUT_DIR, "log_history.csv"), index=False)
tr = hist.dropna(subset=["loss"])[["step", "loss"]] if "loss" in hist else pd.DataFrame(columns=["step", "loss"])
ev = (hist.dropna(subset=["eval_loss"])[["step", "eval_loss"]]
      if "eval_loss" in hist else pd.DataFrame(columns=["step", "eval_loss"]))
print(ev.to_string(index=False))

import matplotlib
matplotlib.use("Agg")  # no blocking window when run as a script
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))
plt.plot(tr.step, tr.loss, label="train loss")
plt.plot(ev.step, ev.eval_loss, "o-", label="validation loss")
plt.xlabel("step"); plt.ylabel("loss"); plt.title(TAG); plt.legend(); plt.grid(alpha=.3)
plt.savefig(os.path.join(OUT_DIR, "loss.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Saved adapter to", OUT_DIR)

# --------------------------------------------------------------------------- #
# 7. Test loss: base model vs adapter (60 test rows)
# --------------------------------------------------------------------------- #
test_rows = data["test"][:]
random.Random(SEED).shuffle(test_rows)
test_ds = Dataset.from_list([{k: r[k] for k in cols} for r in test_rows[:60]])
with_adapter = trainer.evaluate(test_ds, metric_key_prefix="test")["test_loss"]
with model.disable_adapter():
    base = trainer.evaluate(test_ds, metric_key_prefix="test")["test_loss"]
print(f"Test loss  base: {base:.3f} (perplexity {math.exp(base):.1f})")
print(f"Test loss  {TAG}: {with_adapter:.3f} (perplexity {math.exp(with_adapter):.1f})")
with open(os.path.join(OUT_DIR, "test_loss.json"), "w") as f:
    json.dump({"test_loss_base": base, "test_loss_adapter": with_adapter, "test_rows": len(test_ds)}, f, indent=1)

# --------------------------------------------------------------------------- #
# 8. Generate: base vs adapter on 3 test articles
# Inference format: same system message, user turn = instruction + "\n\nNews:\n" + text
# --------------------------------------------------------------------------- #
FastLanguageModel.for_inference(model)  # Unsloth's 2x faster generation
raw_test = load("test")
random.Random(1).shuffle(raw_test)


def generate(instruction, news, max_new_tokens=1024):
    ids = torch.tensor([prompt_ids(instruction, news)], device="cuda")
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=max_new_tokens, max_length=None,  # silences the max_length warning
                             do_sample=False, repetition_penalty=1.05,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids(EOT))
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


samples = []
for r in raw_test[:3]:
    news = tok.decode(tok(r["input"], add_special_tokens=False)["input_ids"][:1200])
    tuned = generate(r["instruction"], news)
    with model.disable_adapter():
        base_out = generate(r["instruction"], news)
    samples.append({"instruction": r["instruction"], "language": r["language"], "depth": r["depth"],
                    "url": r["url"], "reference": r["output"], "base": base_out, "adapter": tuned})
    print("=" * 100, f"\n[{r['language']} / {r['depth']}] {r['url']}")
    print("\n--- BASE ---\n", base_out[:1200])
    print("\n--- ADAPTER ---\n", tuned[:1200])
    print("\n--- REFERENCE ---\n", r["output"][:1200])
pd.DataFrame(samples).to_csv(os.path.join(OUT_DIR, "samples.csv"), index=False, encoding="utf-8-sig")

# --------------------------------------------------------------------------- #
# 9. Package the adapter (standard PEFT adapter, ~170 MB; loads without Unsloth)
# --------------------------------------------------------------------------- #
zip_path = shutil.make_archive(os.path.join(OUT_ROOT, f"adapter_{TAG}"), "zip", OUT_DIR)
if FINISHED:
    shutil.rmtree(CKPT_DIR, ignore_errors=True)  # keep checkpoints only if training must be resumed
print("Packaged:", zip_path, f"({os.path.getsize(zip_path) / 1e6:.0f} MB)", "| finished:", FINISHED)
