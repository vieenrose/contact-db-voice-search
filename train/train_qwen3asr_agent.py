#!/usr/bin/env python3
"""Fine-tune Qwen3-ASR-0.6B as the audio-in agentic tool-caller (the edge alternative to Omni-3B).

Re-elicits the DELIBERATELY de-instruction-tuned Qwen3 decoder to emit Hermes <tool_call> JSON
from SPEECH, multi-turn, while the AuT audio encoder + projector stay FROZEN (reuse the ~40M-hr
alignment). Same dialog data + assistant-span masking as the Omni phase-2/3 trainer; fp16 + LoRA
fits the 8 GB GTX-1070 (no 4-bit needed). Run in the .venv-qa (transformers 4.57.6 + qwen-asr).

  .venv-qa/bin/python train/train_qwen3asr_agent.py --smoke --device cpu        # validate pipeline
  .venv-qa/bin/python train/train_qwen3asr_agent.py --train data/audio/dialogs_phase3_train.jsonl \
      --out runs/qwen3asr-agent --epochs 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
ASR_SR = 16000
IM_START, IM_END = 151644, 151645

SYS = ("You are a phone attendant for a Taiwan office. To find a colleague, call the directory "
       "tool by writing exactly "
       '<tool_call>{"name":"search_contacts","arguments":{"query":"<name as heard>"}}</tool_call>. '
       "If several people share that name, ask which department, then call the tool again adding "
       '"department":"<dept>". After a unique result, connect the caller (name + extension); '
       "if none match, say it was not found. Ignore the caller's own name.")


def load_wav_16k(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(-1)
    if sr != ASR_SR:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=ASR_SR)
    return w


def build_processor():
    return Qwen3ASRProcessor.from_pretrained(MODEL_ID)


def build_model(device="cuda", dtype=torch.float16):
    full = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype)
    thinker = full.thinker
    for p in thinker.audio_tower.parameters():     # freeze the AuT encoder + projector
        p.requires_grad_(False)
    return thinker.to(device)


def make_trainable(thinker):
    thinker.config.use_cache = False
    thinker.gradient_checkpointing_enable()
    thinker.enable_input_require_grads()
    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        # regex scopes LoRA to the DECODER (model.layers.*) only — never the audio_tower.
        target_modules=r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)",
    )
    thinker = get_peft_model(thinker, lora)
    for _, p in thinker.named_parameters():        # fp32 LoRA params for fp16 grad stability
        if p.requires_grad:
            p.data = p.data.float()
    thinker.print_trainable_parameters()
    return thinker


def build_chatml(turns):
    """Dialog turns -> a ChatML string with <|audio_pad|> in user turns + the aligned audio list.
    Tool calls/responses embedded as Hermes TEXT (the ASR tokenizer has no tool template)."""
    parts = [f"<|im_start|>system\n{SYS}<|im_end|>\n"]
    audios = []
    for t in turns:
        if t["role"] == "user":
            audios.append(load_wav_16k(t["audio"]))
            parts.append("<|im_start|>user\n<|audio_pad|><|im_end|>\n")
        elif t["role"] == "assistant" and "tool_call" in t:
            tc = json.dumps({"name": t["tool_call"]["name"], "arguments": t["tool_call"]["arguments"]},
                            ensure_ascii=False)
            parts.append(f"<|im_start|>assistant\n<tool_call>{tc}</tool_call><|im_end|>\n")
        elif t["role"] == "assistant":
            parts.append(f"<|im_start|>assistant\n{t['text']}<|im_end|>\n")
        elif t["role"] == "tool":
            tr = json.dumps(t["content"], ensure_ascii=False)
            parts.append(f"<|im_start|>tool\n<tool_response>{tr}</tool_response><|im_end|>\n")
    return "".join(parts), audios


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
                labels[j] = ids[j]
            i = j + 1
        else:
            i += 1
    return labels


class DialogDS(Dataset):
    def __init__(self, jsonl, processor):
        self.rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
        self.proc = processor
        self.asst_id = processor.tokenizer.convert_tokens_to_ids("assistant")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text, audios = build_chatml(self.rows[i]["turns"])
        enc = self.proc(text=text, audio=audios or None, sampling_rate=ASR_SR, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        enc["labels"] = torch.tensor([mask_assistant(ids, self.asst_id)])
        text_keys = {"input_ids", "attention_mask", "labels"}      # (1,L)->(L,); keep audio dims
        return {k: (v.squeeze(0) if k in text_keys and torch.is_tensor(v) and v.ndim > 1 else v)
                for k, v in enc.items()}


def collate(features, pad_id):
    F = torch.nn.functional
    L = max(f["input_ids"].shape[-1] for f in features)
    batch = {
        "input_ids": torch.stack([F.pad(f["input_ids"], (0, L - f["input_ids"].shape[-1]), value=pad_id) for f in features]),
        "labels": torch.stack([F.pad(f["labels"], (0, L - f["labels"].shape[-1]), value=-100) for f in features]),
    }
    batch["attention_mask"] = (batch["input_ids"] != pad_id).long()
    if "input_features" in features[0]:
        T = max(f["input_features"].shape[-1] for f in features)
        batch["input_features"] = torch.cat([F.pad(f["input_features"], (0, T - f["input_features"].shape[-1])) for f in features])
        for k in ("feature_attention_mask", "audio_feature_lengths"):
            if k in features[0]:
                if features[0][k].ndim >= 2:
                    batch[k] = torch.cat([F.pad(f[k], (0, T - f[k].shape[-1])) for f in features])
                else:
                    batch[k] = torch.cat([f[k] for f in features])
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/audio/dialogs_phase3_train.jsonl")
    ap.add_argument("--out", default="runs/qwen3asr-agent")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    proc = build_processor()
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    model = make_trainable(build_model(device=args.device, dtype=dtype))
    model.config.use_cache = False

    train_path = args.train
    if args.smoke:
        rows = [json.loads(l) for l in open(args.train, encoding="utf-8")]
        mt = [r for r in rows if len(r["turns"]) > 4][:1] + rows[:1]   # 1 multi-turn + 1 single
        mini = Path("runs/_smoke_q3asr.jsonl"); mini.parent.mkdir(parents=True, exist_ok=True)
        mini.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mt))
        train_path = str(mini)

    ds = DialogDS(train_path, proc)
    pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke else args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr, max_steps=2 if args.smoke else -1,
        fp16=(args.device != "cpu"), gradient_checkpointing=True, logging_steps=1,
        save_strategy="no" if args.smoke else "epoch", report_to=[],
        remove_unused_columns=False, label_names=["labels"], use_cpu=(args.device == "cpu"))
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=lambda f: collate(f, pad_id)).train()
    if not args.smoke:
        model.save_pretrained(args.out); proc.save_pretrained(args.out)
    print("SMOKE OK" if args.smoke else f"saved -> {args.out}")


if __name__ == "__main__":
    main()
