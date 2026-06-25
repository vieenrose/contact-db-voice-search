#!/usr/bin/env python3
"""Phase-2: train Qwen-Omni-3B as a multi-turn TOOL-CALLING agent.

Qwen-Omni's chat template has no tool support, so tool-calls/responses are embedded
as Hermes-style TEXT (<tool_call>{...}</tool_call> / <tool_response>...). The model
learns to emit tool-calls + replies across multiple audio turns. Only ASSISTANT turns
are supervised (masked otherwise). batch=1 + QLoRA fits the 8 GB card.

  .venv/bin/python train/train_qwen_agent.py --train data/audio/dialogs_train.jsonl --out runs/agent --epochs 2
  .venv/bin/python train/train_qwen_agent.py --smoke
"""
import argparse, json, sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_qwen import build_model, make_trainable, build_processor
from train_qwen import load_wav_16k, ASR_SR

SYS = ("You are a phone attendant for a Taiwan office. To find a colleague, call the "
       "directory tool by writing exactly "
       '<tool_call>{"name":"search_contacts","arguments":{"query":"<name as heard>"}}</tool_call>. '
       "After the tool result, either connect the caller (give name + extension), ask which "
       "person if several match, or say it was not found. Ignore the caller's own name.")
IM_START, IM_END = 151644, 151645


def build_messages(turns):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]}]
    audios = []
    for t in turns:
        if t["role"] == "user":
            audios.append(load_wav_16k(t["audio"]))
            msgs.append({"role": "user", "content": [{"type": "audio", "audio": None}]})
        elif t["role"] == "assistant" and "tool_call" in t:
            tc = json.dumps({"name": t["tool_call"]["name"], "arguments": t["tool_call"]["arguments"]},
                            ensure_ascii=False)
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"<tool_call>{tc}</tool_call>"}]})
        elif t["role"] == "assistant":
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": t["text"]}]})
        elif t["role"] == "tool":
            tr = json.dumps(t["content"], ensure_ascii=False)
            msgs.append({"role": "tool", "content": [{"type": "text", "text": f"<tool_response>{tr}</tool_response>"}]})
    return msgs, audios


def mask_assistant(ids, asst_id):
    """Supervise tokens inside `<|im_start|>assistant ... <|im_end|>` spans only."""
    labels = [-100] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i] == IM_START and i + 1 < len(ids) and ids[i + 1] == asst_id:
            j = i + 2
            while j < len(ids) and ids[j] != IM_END:
                labels[j] = ids[j]; j += 1
            if j < len(ids):
                labels[j] = ids[j]                       # supervise the closing <|im_end|>
            i = j + 1
        else:
            i += 1
    return labels


class DialogDS(Dataset):
    def __init__(self, jsonl, processor):
        self.rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
        self.proc = processor
        self.asst_id = processor.tokenizer.convert_tokens_to_ids("assistant")

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        msgs, audios = build_messages(self.rows[i]["turns"])
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        enc = self.proc(text=text, audio=audios, sampling_rate=ASR_SR, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        enc["labels"] = torch.tensor([mask_assistant(ids, self.asst_id)])
        text_keys = {"input_ids", "attention_mask", "labels"}   # (1,L) -> (L,)
        return {k: (v.squeeze(0) if k in text_keys and torch.is_tensor(v) and v.ndim > 1 else v)
                for k, v in enc.items()}                          # keep audio (n_audio, …) dims


def collate(features, pad_id):
    F = torch.nn.functional
    L = max(f["input_ids"].shape[-1] for f in features)
    batch = {
        "input_ids": torch.stack([F.pad(f["input_ids"], (0, L - f["input_ids"].shape[-1]), value=pad_id) for f in features]),
        "labels": torch.stack([F.pad(f["labels"], (0, L - f["labels"].shape[-1]), value=-100) for f in features]),
    }
    batch["attention_mask"] = (batch["input_ids"] != pad_id).long()
    if "input_features" in features[0]:        # (n_audio, mel, T) per example — batch=1 friendly
        T = max(f["input_features"].shape[-1] for f in features)
        batch["input_features"] = torch.cat([F.pad(f["input_features"], (0, T - f["input_features"].shape[-1])) for f in features])
        if "feature_attention_mask" in features[0]:
            batch["feature_attention_mask"] = torch.cat(
                [F.pad(f["feature_attention_mask"], (0, T - f["feature_attention_mask"].shape[-1])) for f in features])
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/audio/dialogs_train.jsonl")
    ap.add_argument("--out", default="runs/agent")
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
        rows = [json.loads(l) for l in open(args.train, encoding="utf-8")]
        mt = [r for r in rows if len(r["turns"]) > 4][:1] + rows[:1]   # 1 multi-turn + 1 single
        mini = Path("runs/_smoke_agent.jsonl"); mini.parent.mkdir(parents=True, exist_ok=True)
        mini.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mt))
        train_path = str(mini)

    ds = DialogDS(train_path, proc)
    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke else args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr, max_steps=2 if args.smoke else -1,
        fp16=True, gradient_checkpointing=True, logging_steps=1,
        save_strategy="no" if args.smoke else "epoch",
        report_to=[], remove_unused_columns=False, label_names=["labels"])
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=lambda f: collate(f, pad_id)).train()
    if not args.smoke:
        model.save_pretrained(args.out); proc.save_pretrained(args.out)
    print("OK" if args.smoke else f"saved -> {args.out}")


if __name__ == "__main__":
    main()
