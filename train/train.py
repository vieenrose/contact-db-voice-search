#!/usr/bin/env python3
"""Train Ultravox (whisper-base encoder) on the attendant task: audio -> {action,name}.

Freezes encoder + Llama-1B, trains the fresh projector + a LoRA on the LLM (see
build.py). batch=1 + grad-accum + fp16 + grad-checkpointing fit the 8 GB GTX 1070.
The LLM/encoder are frozen so the model learns the audio->name/action mapping while
keeping whisper-base's real-speaker robustness.

  .venv/bin/python train/train.py --train data/audio/train.jsonl --val data/audio/val.jsonl \
      --out runs/v1 --epochs 2
  .venv/bin/python train/train.py --smoke    # 2 steps on a tiny set to validate the pipeline
"""
import argparse, json, sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import build_model, make_trainable, build_processor, AUDIO_MODEL

ASR_SR = 16000
SYS = ("You are a phone attendant for a Taiwan office. Identify the colleague the "
       "caller is asking for. Respond ONLY with JSON: "
       '{"action":"resolve","name":"<English name>"} when you recognize the person, '
       '{"action":"clarify",...} if under-specified, or {"action":"not_found"}.')


def load_wav_16k(path):
    x, sr = sf.read(path); x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(-1)
    if sr != ASR_SR:
        fr = Fraction(ASR_SR, sr).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    return x


class AudioDS(Dataset):
    def __init__(self, jsonl, processor):
        self.rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
        self.proc = processor
        self.tok = processor.tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav = load_wav_16k(r["audio"])
        target = r["target_text"]
        turns = [{"role": "system", "content": SYS},
                 {"role": "user", "content": "<|audio|>"},
                 {"role": "assistant", "content": target}]
        full = self.tok.apply_chat_template(turns, tokenize=False)
        prompt = self.tok.apply_chat_template(turns[:-1], tokenize=False, add_generation_prompt=True)
        ex = self.proc(text=full, audio=wav, sampling_rate=ASR_SR, return_tensors="pt")
        pr = self.proc(text=prompt, audio=wav, sampling_rate=ASR_SR, return_tensors="pt")
        plen = pr["input_ids"].shape[1]
        labels = ex["input_ids"].clone()
        labels[:, :plen] = -100
        ex["labels"] = labels
        return {k: (v.squeeze(0) if torch.is_tensor(v) and v.ndim and v.shape[0] == 1 else v)
                for k, v in ex.items()}


def collate(features, pad_id):
    """Pad input_ids/labels (token dim) and audio_values (frame dim); stack audio scalars."""
    F = torch.nn.functional
    L = max(f["input_ids"].shape[-1] for f in features)
    T = max(f["audio_values"].shape[-1] for f in features)
    batch = {
        "input_ids": torch.stack([F.pad(f["input_ids"], (0, L - f["input_ids"].shape[-1]), value=pad_id) for f in features]),
        "labels": torch.stack([F.pad(f["labels"], (0, L - f["labels"].shape[-1]), value=-100) for f in features]),
        "audio_values": torch.stack([F.pad(f["audio_values"], (0, T - f["audio_values"].shape[-1])) for f in features]),
    }
    batch["attention_mask"] = (batch["input_ids"] != pad_id).long()
    for k in ("audio_lens", "audio_token_len", "audio_token_start_idx", "audio_batch_size"):
        if k in features[0]:
            batch[k] = torch.stack([f[k].reshape(()) for f in features]).reshape(-1)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/audio/train.jsonl")
    ap.add_argument("--val", default="data/audio/val.jsonl")
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stock", action="store_true",
                    help="stock large encoder + pretrained projector (no whisper-base swap)")
    args = ap.parse_args()

    audio_model = None if args.stock else AUDIO_MODEL
    processor = build_processor(audio_model)
    model, _ = build_model(audio_model)
    model = make_trainable(model)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_path = args.train
    if args.smoke:
        # tiny set from whatever base wavs exist, to validate forward/backward
        base = sorted(Path("data/audio/base").glob("e*.wav"))[:2]
        mini = Path("runs/_smoke.jsonl"); mini.parent.mkdir(parents=True, exist_ok=True)
        with mini.open("w") as f:
            for w in base:
                f.write(json.dumps({"audio": str(w),
                                    "target_text": '{"action":"resolve","name":"Ray Hsu"}'}) + "\n")
        train_path = str(mini)

    train_ds = AudioDS(train_path, processor)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke else args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr,
        max_steps=2 if args.smoke else -1,
        fp16=True, gradient_checkpointing=True, logging_steps=1,
        save_strategy="no" if args.smoke else "epoch",
        report_to=[], remove_unused_columns=False, label_names=["labels"],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=lambda f: collate(f, pad_id))
    trainer.train()
    if not args.smoke:
        trainer.save_model(args.out)
        processor.save_pretrained(args.out)
    print("OK" if args.smoke else f"saved -> {args.out}")


if __name__ == "__main__":
    main()
