#!/usr/bin/env python3
"""Build Qwen2.5-Omni-3B for QLoRA fine-tuning (audio -> {action,name}).

4-bit base (fits the 8 GB GTX 1070), LoRA on the thinker's text model only — the
audio encoder (which hears Mandarin correctly) stays frozen. No projector to train
from scratch: Qwen's audio->text alignment is integrated + pretrained (v2-style).
"""
import torch
from transformers import (Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-Omni-3B"
# We train the THINKER directly (the Omni wrapper has no usable forward). Within the
# thinker, the text LLM is `model.*`; LoRA targets its attention only (audio_tower frozen).
LORA_RE = r"model\..*\.(q_proj|k_proj|v_proj|o_proj)"


def build_model(dtype=torch.float16):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=bnb, attn_implementation="sdpa",
        device_map="cuda", enable_audio_output=False)
    return omni.thinker          # the audio->text model with the real forward()


def make_trainable(model, r=16, alpha=32, grad_ckpt=True):
    model.config.use_cache = False
    model.enable_input_require_grads()       # grads must reach inputs for grad-ckpt + frozen base
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
                      target_modules=LORA_RE, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def build_processor():
    return Qwen2_5OmniProcessor.from_pretrained(MODEL)
