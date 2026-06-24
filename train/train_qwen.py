#!/usr/bin/env python3
"""QLoRA fine-tune Qwen2.5-Omni-3B on the attendant task (audio -> {action,name}).

  .venv/bin/python train/train_qwen.py --train data/audio/train.jsonl --out runs/qwen --epochs 2
  .venv/bin/python train/train_qwen.py --smoke
"""
import argparse, json, sys
from fractions import Fraction
from pathlib import Path

import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_qwen import build_model, make_trainable, build_processor

ASR_SR = 16000
SYS = ("You are a phone attendant for a Taiwan office. Identify the colleague the caller "
       "is asking for and respond ONLY with JSON: "
       '{"action":"resolve","name":"<English name>"}, {"action":"clarify",...}, or {"action":"not_found"}.')


def load_wav_16k(path):
    x, sr = sf.read(path); x = np.asarray(x, np.float32)
    if x.ndim > 1: x = x.mean(-1)
    if sr != ASR_SR:
        fr = Fraction(ASR_SR, sr).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    return x


def convo(target=None):
    c = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
         {"role": "user", "content": [{"type": "audio", "audio": None}]}]
    if target is not None:
        c.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return c


class AudioDS(Dataset):
    def __init__(self, jsonl, processor):
        self.rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
        self.proc = processor

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav = load_wav_16k(r["audio"])
        full = self.proc.apply_chat_template(convo(r["target_text"]), add_generation_prompt=False, tokenize=False)
        prompt = self.proc.apply_chat_template(convo(), add_generation_prompt=True, tokenize=False)
        ex = self.proc(text=full, audio=[wav], sampling_rate=ASR_SR, return_tensors="pt")
        pr = self.proc(text=prompt, audio=[wav], sampling_rate=ASR_SR, return_tensors="pt")
        plen = pr["input_ids"].shape[1]
        labels = ex["input_ids"].clone(); labels[:, :plen] = -100
        ex["labels"] = labels
        return {k: (v.squeeze(0) if torch.is_tensor(v) and v.shape[0] == 1 else v) for k, v in ex.items()}


def collate(features, pad_id):
    F = torch.nn.functional
    L = max(f["input_ids"].shape[-1] for f in features)
    batch = {
        "input_ids": torch.stack([F.pad(f["input_ids"], (0, L - f["input_ids"].shape[-1]), value=pad_id) for f in features]),
        "labels": torch.stack([F.pad(f["labels"], (0, L - f["labels"].shape[-1]), value=-100) for f in features]),
    }
    batch["attention_mask"] = (batch["input_ids"] != pad_id).long()
    # audio features (mel): pad on time dim; carry the feature attention mask
    if "input_features" in features[0]:
        T = max(f["input_features"].shape[-1] for f in features)
        batch["input_features"] = torch.stack([F.pad(f["input_features"], (0, T - f["input_features"].shape[-1])) for f in features])
        if "feature_attention_mask" in features[0]:
            batch["feature_attention_mask"] = torch.stack(
                [F.pad(f["feature_attention_mask"], (0, T - f["feature_attention_mask"].shape[-1])) for f in features])
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/audio/train.jsonl")
    ap.add_argument("--out", default="runs/qwen")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    proc = build_processor()
    model = make_trainable(build_model())
    model.config.use_cache = False

    train_path = args.train
    if args.smoke:
        base = sorted(Path("data/audio/base").glob("e*.wav"))[:2]
        mini = Path("runs/_smoke_qwen.jsonl"); mini.parent.mkdir(parents=True, exist_ok=True)
        with mini.open("w") as f:
            for w in base:
                f.write(json.dumps({"audio": str(w), "target_text": '{"action":"resolve","name":"Ray Hsu"}'}) + "\n")
        train_path = str(mini)

    ds = AudioDS(train_path, proc)
    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke else args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr, max_steps=2 if args.smoke else -1,
        fp16=True, gradient_checkpointing=True, logging_steps=1,
        save_strategy="no" if args.smoke else "epoch",
        report_to=[], remove_unused_columns=False, label_names=["labels"])
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=lambda f: collate(f, pad_id)).train()
    if not args.smoke:
        model.save_pretrained(args.out); proc.save_pretrained(args.out)
    print("OK" if args.smoke else f"saved -> {args.out}")


if __name__ == "__main__":
    main()
