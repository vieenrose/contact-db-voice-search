#!/usr/bin/env python3
"""Build the trainable Ultravox model with a whisper-BASE encoder (for gen1 latency).

Swap: set config.audio_model_id=openai/whisper-base + matching audio_config, so
_create_audio_tower loads whisper-base (80 mel, d_model 512) and the projector
re-inits to 512*stack_factor inputs. Encoder + Llama-1B are loaded pretrained and
FROZEN; we train only the fresh projector + a LoRA on the LLM.

LoRA is scoped by regex to language_model.* — the whisper encoder also has
q/k/v/o_proj and must stay frozen.

Run standalone to validate construction + trainable-param budget on the GPU:
    .venv/bin/python train/build.py
"""
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, WhisperConfig, WhisperFeatureExtractor
from peft import LoraConfig, get_peft_model

BASE = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
AUDIO_MODEL = "openai/whisper-base"
TEXT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LLM_LORA_RE = r"language_model\..*\.(q_proj|k_proj|v_proj|o_proj)"


def build_model(audio_model=AUDIO_MODEL, dtype=torch.float16):
    cfg = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
    cfg.audio_model_id = audio_model
    cfg.audio_config = WhisperConfig.from_pretrained(audio_model)   # 80 mel / d_model 512
    cfg.text_model_id = TEXT_MODEL
    model = AutoModel.from_config(cfg, trust_remote_code=True)       # loads pretrained sub-models
    return model.to(dtype), cfg


def make_trainable(model, r=16, alpha=32):
    for p in model.audio_tower.parameters():
        p.requires_grad = False
    for p in model.language_model.parameters():
        p.requires_grad = False
    lora = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
                      target_modules=LLM_LORA_RE,
                      modules_to_save=["multi_modal_projector"])
    model = get_peft_model(model, lora)
    # fp16 AMP requires trainable params in fp32 (grad-scaler can't unscale fp16 grads;
    # Pascal has no bf16). Frozen encoder + LLM stay fp16 to save the 8 GB card.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def build_processor(audio_model=AUDIO_MODEL):
    """UltravoxProcessor with whisper-BASE feature extractor (80 mel) + Llama tokenizer."""
    proc = transformers.AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    proc.audio_processor = WhisperFeatureExtractor.from_pretrained(audio_model)
    return proc


if __name__ == "__main__":
    print("transformers", transformers.__version__)
    model, cfg = build_model()
    print("audio:", cfg.audio_config.num_mel_bins, "mel /", cfg.audio_config.d_model, "d_model")
    for n, m in model.named_modules():
        if "projector" in n and isinstance(m, torch.nn.Linear):
            print(f"  {n}: {m.in_features} -> {m.out_features}")
    model = make_trainable(model)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    print(f"trainable {tr/1e6:.2f}M / total {tot/1e6:.1f}M  ({100*tr/tot:.2f}%)")
    print("trainable module groups:")
    seen = set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            key = "projector" if "projector" in n else ("lora" if "lora" in n else n.split(".")[0])
            if key not in seen:
                seen.add(key); print("   -", key)
