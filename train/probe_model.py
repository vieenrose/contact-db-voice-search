#!/usr/bin/env python3
"""Probe Ultravox architecture before writing the trainer.

Answers: (1) does it load under transformers 5.12.1 trust_remote_code? (2) audio
encoder type + mel bins + hidden size (matters for the whisper-base swap), (3)
projector in/out dims, (4) module names to freeze / LoRA-target, (5) param counts.
"""
import torch, transformers
from transformers import AutoProcessor, AutoModel

MODEL = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
print("transformers", transformers.__version__)

proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
print("processor:", type(proc).__name__)
fe = getattr(proc, "audio_processor", None) or getattr(proc, "feature_extractor", None)
print("feature extractor:", type(fe).__name__ if fe else None,
      "| n_mels:", getattr(fe, "feature_size", getattr(fe, "num_mel_bins", "?")),
      "| sr:", getattr(fe, "sampling_rate", "?"))

model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float16)
print("model:", type(model).__name__)

cfg = model.config
print("\n=== config ===")
for k in ("audio_model_id", "text_model_id", "hidden_size", "stack_factor",
          "projector_act", "audio_latency_block_size"):
    print(f"  {k}: {getattr(cfg, k, None)}")
ac = getattr(cfg, "audio_config", None)
if ac is not None:
    print("  audio_config:", type(ac).__name__,
          "num_mel_bins:", getattr(ac, "num_mel_bins", "?"),
          "d_model:", getattr(ac, "d_model", getattr(ac, "hidden_size", "?")))

print("\n=== top-level submodules ===")
for name, mod in model.named_children():
    n = sum(p.numel() for p in mod.parameters())
    print(f"  {name:24} {type(mod).__name__:30} {n/1e6:8.1f}M params")

# locate projector + show its linear shapes
print("\n=== projector linears ===")
for name, mod in model.named_modules():
    if "project" in name.lower() and isinstance(mod, torch.nn.Linear):
        print(f"  {name}: {mod.in_features} -> {mod.out_features}")

print("\nTOTAL params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
