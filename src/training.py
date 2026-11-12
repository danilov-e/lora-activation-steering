"""Ordinary LoRA and response-only supervised fine-tuning."""
import torch
from peft import LoraConfig, TaskType, get_peft_model


def encode_response(tokenizer, row, max_length):
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:1], tokenize=True, add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    )
    if full[:len(prompt)] != prompt:
        raise ValueError(f"{row['id']}: inconsistent assistant boundary.")
    response = full[len(prompt):]
    eos = tokenizer.eos_token_id
    if not response or response[0] == eos or response.count(eos) != 1:
        raise ValueError(f"{row['id']}: expected an answer followed by one EOS.")
    response = response[:response.index(eos) + 1]
    ids = prompt + response
    if len(ids) > max_length:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceed max_length={max_length}.")
    return {"input_ids": ids, "labels": [-100] * len(prompt) + response}


def collate_responses(examples, *, pad_id):
    width = max(len(row["input_ids"]) for row in examples)
    ids, masks, labels = [], [], []
    for row in examples:
        n = len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * (width - n))
        masks.append([1] * n + [0] * (width - n))
        labels.append(row["labels"] + [-100] * (width - n))
    return {
        key: torch.tensor(value, dtype=torch.long)
        for key, value in (("input_ids", ids), ("attention_mask", masks), ("labels", labels))
    }


def attach_lora(base, settings, revision):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, bias="none",
        use_rslora=False, use_dora=False, **settings,
    )
    model = get_peft_model(base, config, revision=revision)
    model.config.use_cache = False
    model.train()
    return model
