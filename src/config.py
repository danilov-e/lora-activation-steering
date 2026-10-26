"""Read the experiment settings; reject misspellings and invalid values."""

import math
import re
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a complete YAML config."""
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    fields = {
        "model": {"id", "revision", "device", "dtype"},
        "run": {"seed", "batch_size"},
        "generation": {
            "max_new_tokens", "min_new_tokens", "do_sample", "temperature",
            "top_p", "top_k", "repetition_penalty", "use_cache",
        },
    }
    if not isinstance(config, dict) or set(config) != set(fields):
        raise ValueError("Config must contain model, run and generation sections.")
    for section, expected in fields.items():
        values = config[section]
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f"{section}: expected keys {sorted(expected)}")

    model, run, gen = config["model"], config["run"], config["generation"]
    if not isinstance(model["id"], str) or not model["id"].strip():
        raise ValueError("model.id must be a nonempty string.")
    if not isinstance(model["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("model.revision must be a full 40-character commit SHA.")
    if model["device"] not in ("auto", "cuda", "cpu"):
        raise ValueError("model.device must be auto, cuda or cpu.")
    if model["dtype"] not in ("auto", "float16", "float32"):
        raise ValueError("model.dtype must be auto, float16 or float32.")
    for name, value, minimum in [
        ("seed", run["seed"], 0), ("batch_size", run["batch_size"], 1),
        ("max_new_tokens", gen["max_new_tokens"], 1),
        ("min_new_tokens", gen["min_new_tokens"], 0), ("top_k", gen["top_k"], 0),
    ]:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if run["seed"] >= 2**32 or gen["min_new_tokens"] > gen["max_new_tokens"]:
        raise ValueError("Invalid seed or token limits.")
    for name in ("temperature", "top_p", "repetition_penalty"):
        value = gen[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if gen["top_p"] > 1 or any(type(gen[k]) is not bool for k in ("do_sample", "use_cache")):
        raise ValueError("top_p must be <= 1; do_sample/use_cache must be booleans.")
    return config
