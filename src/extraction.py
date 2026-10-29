"""Extract response-averaged vectors from Qwen2 decoder block outputs."""

import torch


def validate_pairs(rows: list[dict]) -> list[str]:
    """Require complete, unique extraction pairs for one trait; preserve pair order."""
    if not rows:
        raise ValueError("No selected answers.")
    groups, ids, traits = {}, set(), set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("Answer ids must be nonempty and unique.")
        ids.add(row["id"])
        if row.get("split") != "extract" or not isinstance(row.get("trait"), str):
            raise ValueError("Expected extraction answers with a trait.")
        traits.add(row["trait"])
        if not isinstance(row.get("pair_id"), str) or not row["pair_id"]:
            raise ValueError("Every answer needs a pair_id.")
        polarity = row.get("polarity")
        if polarity not in ("pos", "neg"):
            raise ValueError("polarity must be pos or neg.")
        for key in ("prompt", "answer", "question"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Expected a nonempty {key}.")
        pair = groups.setdefault(row["pair_id"], {})
        if polarity in pair:
            raise ValueError("Duplicate pair polarity.")
        pair[polarity] = row
    if len(traits) != 1:
        raise ValueError("Expected exactly one trait.")
    for pair in groups.values():
        if set(pair) != {"pos", "neg"}:
            raise ValueError("Incomplete pair.")
        for key in ("question_id", "instruction_id", "repeat", "question"):
            if key not in pair["pos"] or pair["pos"][key] != pair["neg"].get(key):
                raise ValueError(f"Pair metadata mismatch: {key}.")
    return list(groups)


@torch.inference_mode()
def response_means(model, tokenizer, rows: list[dict]) -> torch.Tensor:
    """Return CPU float32 [answers, blocks, hidden], excluding prompt and padding.

    As in Persona Vectors, tokenize prompt + decoded answer without an appended
    EOS. Row k of the layer axis is model.model.layers[k]'s output, before the
    final model norm. Each answer has equal weight regardless of its length.
    """
    if model.training or model.config.model_type != "qwen2":
        raise ValueError("Expected a Qwen2 causal LM in eval mode.")
    if not rows or tokenizer.pad_token_id is None:
        raise ValueError("Expected nonempty rows and a tokenizer padding token.")
    lengths = []
    texts = []
    for row in rows:
        prompt, answer = row["prompt"], row["answer"]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
        if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Tokenization changed the prompt/answer boundary.")
        if len(full_ids) <= len(prompt_ids):
            raise ValueError("The answer has no tokens.")
        if "prompt_token_ids" in row and row["prompt_token_ids"] != prompt_ids:
            raise ValueError("Saved prompt tokens do not match the tokenizer.")
        if len(full_ids) > model.config.max_position_embeddings:
            raise ValueError("Prompt and answer exceed the model context length.")
        lengths.append(len(prompt_ids))
        texts.append(prompt + answer)
    inputs = tokenizer(texts, padding=True, add_special_tokens=False,
                       return_tensors="pt").to(model.device)
    mask = inputs.attention_mask.bool()
    positions = inputs.attention_mask.cumsum(-1) - 1
    response_mask = mask & (positions >= torch.tensor(lengths, device=model.device)[:, None])
    inputs["position_ids"] = positions.masked_fill(~mask, 0)
    means = {}

    def capture(index):
        def hook(module, arguments, output):
            hidden = output[0] if isinstance(output, tuple) else output
            means[index] = torch.stack([
                hidden[i, response_mask[i]].float().mean(0).cpu()
                for i in range(len(rows))
            ])
        return hook

    handles = []
    try:
        for index, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(capture(index)))
        model.model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    result = torch.stack([means[k] for k in range(len(model.model.layers))], dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite response activations.")
    return result


def extract_vector(model, tokenizer, rows: list[dict], *, batch_size: int = 1) -> dict:
    """Average answers within each polarity, then subtract negative from positive."""
    pair_ids = validate_pairs(rows)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    sums, counts = {}, {"pos": 0, "neg": 0}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        means = response_means(model, tokenizer, batch)
        for row, mean in zip(batch, means):
            polarity = row["polarity"]
            if polarity not in sums:
                sums[polarity] = torch.zeros_like(mean, dtype=torch.float64)
            sums[polarity] += mean.double()
            counts[polarity] += 1
        print(f"Extracted {start + len(batch)}/{len(rows)} answers", flush=True)
    positive = (sums["pos"] / counts["pos"]).float()
    negative = (sums["neg"] / counts["neg"]).float()
    return {"vectors": positive - negative, "positive_mean": positive,
            "negative_mean": negative, "pair_ids": pair_ids,
            "layer_indices": list(range(len(model.model.layers))),
            "layer_site": "decoder_block_output_before_final_norm",
            "trait": rows[0]["trait"], "n_pairs": len(pair_ids)}
