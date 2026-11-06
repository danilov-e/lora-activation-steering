"""Generate one batch; the caller controls batch composition and saving."""

import torch
from transformers import set_seed

from src.records import generation_messages, user_messages


def _render_chat_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_prompt(tokenizer, question: str, system: str | None = None) -> str:
    """Render a single-turn prompt, delegating the default system text to the template."""
    return _render_chat_prompt(tokenizer, user_messages(question, system))


@torch.inference_mode()
def generate_batch(model, tokenizer, jobs: list[dict], settings: dict, *, seed: int) -> list[dict]:
    """Return one record per job, including the exact prompt and sampled tokens.

    Chat jobs carry messages, a unique id and a question group id.
    Reproducibility assumes the same model, settings, batch and environment.
    The seed resets the process RNGs via transformers.set_seed.
    """
    if not jobs:
        return []
    if model.training or tokenizer.padding_side != "left":
        raise ValueError("Generation requires model.eval() and left padding.")
    prompts = [_render_chat_prompt(tokenizer, messages) for messages in generation_messages(jobs)]
    inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    # Every generated sequence starts after the SAME padded input width.
    prompt_width = inputs.input_ids.shape[1]
    eos = model.generation_config.eos_token_id
    if eos is None:
        raise ValueError("The model generation config must define EOS.")
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    sampling = dict(settings)
    if not sampling["do_sample"]:
        for key in ("temperature", "top_p", "top_k"):
            sampling.pop(key)
        # Neutralize checkpoint sampling defaults for greedy decoding.
        sampling.update(temperature=1.0, top_p=1.0, top_k=50)
    set_seed(seed)
    output = model.generate(
        **inputs, **sampling, pad_token_id=tokenizer.pad_token_id,
        num_beams=1, num_return_sequences=1, return_dict_in_generate=False,
    )
    rows = []
    for i, (job, prompt) in enumerate(zip(jobs, prompts)):
        tokens = output[i, prompt_width:].tolist()
        eos_at = next((j for j, token in enumerate(tokens) if token in eos_ids), None)
        if eos_at is not None:
            tokens = tokens[:eos_at + 1]  # keep EOS, discard padding after it
        prompt_ids = inputs.input_ids[i][inputs.attention_mask[i].bool()].tolist()
        rows.append({
            **job, "prompt": prompt, "prompt_token_ids": prompt_ids,
            "answer": tokenizer.decode(tokens, skip_special_tokens=True),
            "generated_token_ids": tokens, "generated_tokens": len(tokens),
            "finish_reason": "eos" if eos_at is not None else "length",
            "batch_seed": seed, "batch_job_ids": [item["id"] for item in jobs],
        })
    return rows
