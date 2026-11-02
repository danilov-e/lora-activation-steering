"""Prepare response-only Qwen SFT batches and attach a trainable LoRA adapter."""

import hashlib
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model


def load_prepared_splits(directory: Path) -> tuple[dict, dict]:
    """Read the normal split, verifying the prepared bytes, counts and group separation."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config"]["source"]["member"] != "dataset/sycophancy/normal.jsonl":
        raise ValueError("Expected the prepared sycophancy/normal dataset.")
    splits, groups, seen_ids = {}, {}, set()
    for name in ("train", "validation"):
        filename = f"{name}.jsonl"
        spec = manifest["files"][filename]
        raw = (directory / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {filename}.")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        if not rows or len(rows) != spec["n_records"]:
            raise ValueError(f"Record count mismatch for {filename}.")
        groups[name] = set()
        for row in rows:
            if row["split"] != name or row["id"] in seen_ids:
                raise ValueError(f"Wrong split or duplicate id in {filename}.")
            seen_ids.add(row["id"])
            groups[name].add(row["group_id"])
        splits[name] = rows
    if groups["train"] & groups["validation"]:
        raise ValueError("A question group occurs in both splits.")
    return splits, manifest


def encode_response(tokenizer, row: dict, max_length: int) -> dict:
    """Mask the complete prompt; supervise the assistant content and its closing EOS."""
    messages = row["messages"]
    if [m["role"] for m in messages] != ["user", "assistant"] or any(
        not isinstance(m["content"], str) or not m["content"].strip() for m in messages
    ):
        raise ValueError(f"{row['id']}: expected one user message and one assistant answer.")
    prompt = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError(f"{row['id']}: assistant boundary does not match the Qwen template.")
    # Qwen appends a newline after <|im_end|>. Only the content and <|im_end|> are targets.
    response = full[len(prompt):]
    eos = tokenizer.eos_token_id
    if response.count(eos) != 1 or response[0] == eos:
        raise ValueError(f"{row['id']}: expected assistant tokens followed by one EOS.")
    response = response[:response.index(eos) + 1]
    ids = prompt + response
    if len(ids) > max_length:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceed max_length={max_length}.")
    return {"input_ids": ids, "labels": [-100] * len(prompt) + response}


def collate_responses(examples: list[dict], *, pad_id: int) -> dict:
    """Right-pad by position. A real EOS keeps its label even when EOS is also PAD."""
    width = max(len(x["input_ids"]) for x in examples)
    ids, labels, masks = [], [], []
    for example in examples:
        n = len(example["input_ids"])
        ids.append(example["input_ids"] + [pad_id] * (width - n))
        masks.append([1] * n + [0] * (width - n))
        labels.append(example["labels"] + [-100] * (width - n))
    return {key: torch.tensor(value, dtype=torch.long) for key, value in
            (("input_ids", ids), ("attention_mask", masks), ("labels", labels))}


def attach_lora(base, settings: dict, revision: str):
    """PEFT freezes the base. Keep trainable adapter weights FP32 for AMP on T4."""
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, bias="none", **settings)
    targets = set(settings["target_modules"])
    available = {name.rsplit(".", 1)[-1] for name, _ in base.named_modules()}
    if targets - available:
        raise ValueError(f"Missing LoRA targets: {sorted(targets - available)}")
    model = get_peft_model(base, config, revision=revision)
    model.config.use_cache = False
    model.train()
    return model
