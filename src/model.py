"""Load model and tokenizer from the same pinned checkpoint."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device_and_dtype(device: str = "auto", dtype: str = "auto") -> tuple:
    """Choose one device. Auto uses FP16 on CUDA and FP32 on CPU."""
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable.")
    if dtype == "auto":
        dtype = "float16" if device == "cuda" else "float32"
    if dtype not in ("float16", "float32"):
        raise ValueError("dtype must be auto, float16 or float32.")
    if device == "cpu" and dtype == "float16":
        raise ValueError("This initial CPU configuration uses float32; select auto or float32.")
    return torch.device(device), getattr(torch, dtype)


def load_model_and_tokenizer(settings: dict) -> tuple:
    """Return a decoder-only model in eval mode and a left-padding tokenizer."""
    device, dtype = resolve_device_and_dtype(settings["device"], settings["dtype"])
    checkpoint = dict(revision=settings["revision"], trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(settings["id"], **checkpoint, use_fast=True)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError("This project requires an EOS token and a chat template.")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        settings["id"], **checkpoint, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    if model.config.is_encoder_decoder:
        raise ValueError("This project expects a decoder-only causal language model.")
    model.eval()
    return model, tokenizer
