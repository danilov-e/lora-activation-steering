"""Run setup, vector extraction, generation and ordinary LoRA."""
import gc
import hashlib
import io
import json
import math
import random
import re
import time
from contextlib import contextmanager, nullcontext
from functools import partial
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.artifacts import read_json, read_rows, sha, write_bytes, write_json
from src.judge import DEFAULT_MODEL, load_ratings
from src.model import format_prompt, generate_answer, load_model_and_tokenizer
from src.steering import steer_from, steer_response, steer_training
from src.training import attach_lora, collate_responses, encode_response


RUN_FORMAT = 4
SPLITS = ("extract", "lora", "dev", "test")


def default_config():
    return {
        "model": {
            "id": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "device": "auto",
            "dtype": "auto",
        },
        "seed": 42,
        "layers": [18, 19, 20, 21],
        "alphas": [1, 2, 2.5, 3],
        "dev_repeats": 1,
        "test_repeats": 3,
        "generation": {
            "max_new_tokens": 256, "min_new_tokens": 1, "do_sample": True,
            "temperature": 0.7, "top_p": 0.9, "top_k": 0,
            "repetition_penalty": 1.0, "use_cache": True,
        },
        "lora": {
            "r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "training": {
            "epochs": 3, "learning_rate": 1e-4, "batch_size": 1,
            "gradient_accumulation_steps": 4, "max_length": 512, "validation_fraction": 0.2,
        },
        "judge": {
            "model": DEFAULT_MODEL, "reasoning": "none", "batch_size": 5, "max_output_tokens": 4096,
        },
        "instruction": "\n\nAnswer directly and briefly, in at most four sentences.",
    }


def code_hash(project):
    source = Path(project) / "src"
    files = ("artifacts.py", "workflow.py", "model.py", "steering.py", "training.py")
    hashes = "".join(sha(source / file) for file in files)
    return hashlib.sha256(hashes.encode()).hexdigest()


def _run_path(project, name, trait, runs_dir):
    if trait not in ("sycophancy", "evil", "hallucination"):
        raise ValueError(f"Unknown trait: {trait}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("RUN_NAME must use letters, digits, hyphens or underscores.")
    return Path(runs_dir or project / "runs") / name / trait


def _validate_config(config):
    layers, alphas = config["layers"], config["alphas"]
    if not layers or any(type(x) is not int or x < 0 for x in layers):
        raise ValueError("Enter nonnegative integer layer indices.")
    if not alphas or any(type(x) not in (float, int) or not math.isfinite(x) or x == 0 for x in alphas):
        raise ValueError("Enter finite nonzero alpha values. Baseline is added automatically.")
    if len(layers) != len(set(layers)) or len(alphas) != len(set(alphas)):
        raise ValueError("Layers and alpha values must be unique.")
    if any(type(config[key]) is not int or config[key] < 1 for key in ("dev_repeats", "test_repeats")):
        raise ValueError("Repeat counts must be positive integers.")
    if not 0 < config["training"]["validation_fraction"] < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")


def create_run(project, name, trait, config=None, runs_dir=None, git_commit=None):
    project = Path(project)
    root = _run_path(project, name, trait, runs_dir)
    config = json.loads(json.dumps(config if config is not None else default_config(), allow_nan=False))
    _validate_config(config)
    if (root / "run.json").exists():
        open_run(project, name, trait, runs_dir=runs_dir)
        if run_info(root)["config"] != config:
            raise ValueError(f"{root}: settings are frozen. Use a new RUN_NAME to change them.")
        return root

    data, snapshots = {}, {}
    ids, questions = set(), set()
    required = {"id", "question", "pos", "neg"}
    for split in SPLITS:
        source = project / "data" / trait / (split + ".jsonl")
        raw = source.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(rows) < 2:
            raise ValueError(f"{source}: expected at least two pairs.")
        for row in rows:
            if set(row) != required or not all(isinstance(x, str) and x.strip() for x in row.values()):
                raise ValueError(f"{source}: each row needs nonempty id, question, pos and neg.")
            question = " ".join(row["question"].casefold().split())
            if row["id"] in ids or question in questions or row["pos"].strip() == row["neg"].strip():
                raise ValueError(f"{source}: duplicate pair/question or identical answers: {row['id']}")
            ids.add(row["id"])
            questions.add(question)
        data[split] = {"sha256": hashlib.sha256(raw).hexdigest(), "pairs": len(rows)}
        snapshots[split] = raw

    for split, raw in snapshots.items():
        write_bytes(root / "inputs" / (split + ".jsonl"), raw)
    write_json(root / "run.json", {
        "format_version": RUN_FORMAT, "trait": trait, "config": config,
        "data": data, "code_sha256": code_hash(project), "git_commit": git_commit,
    })
    write_json(root / "environment.json", {
        name: version(name) for name in ("torch", "transformers", "peft", "accelerate", "openai")
    })
    return root


def open_run(project, name, trait, runs_dir=None):
    project = Path(project)
    root = _run_path(project, name, trait, runs_dir)
    info = run_info(root)
    if info.get("format_version") != RUN_FORMAT:
        raise ValueError(f"{root}: historical run. Use its saved notebooks or a new RUN_NAME.")
    if info["trait"] != trait or info["code_sha256"] != code_hash(project):
        raise ValueError(f"{root}: the run expects a different project revision.")
    for split in SPLITS:
        input_path = root / "inputs" / (split + ".jsonl")
        if not input_path.is_file() or sha(input_path) != info["data"][split]["sha256"]:
            raise ValueError(f"{input_path}: restore the original input snapshot.")
    return root


def run_info(root):
    return read_json(Path(root) / "run.json")


def pairs(root, split):
    return read_rows(Path(root) / "inputs" / (split + ".jsonl"))


def _vector_path(root, label):
    return Path(root) / "vectors" / label / "vectors.pt"


def load_vector(root, label):
    return torch.load(_vector_path(root, label), map_location="cpu", weights_only=True)


def _adapter_dir(root, label):
    return Path(root) / "adapters" / label


def _check_adapter(root, label):
    directory = _adapter_dir(root, label)
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run notebook 02 first.")
    manifest = read_json(path)
    weights = directory / "adapter" / "adapter_model.safetensors"
    config = directory / "adapter" / "adapter_config.json"
    if manifest["status"] != "complete":
        raise ValueError(f"{directory}: finish training in notebook 02.")
    if not weights.is_file() or not config.is_file() or sha(weights) != manifest["adapter_sha256"]:
        raise ValueError(f"{directory}: restore the final adapter matching the training manifest.")
    return manifest["adapter_sha256"]


def _checked_vector(root, label, adapter_hash=None):
    path = _vector_path(root, label)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: run notebook {'01' if label == 'base' else '02'} first.")
    bundle = load_vector(root, label)
    expected = {
        "extract_sha256": run_info(root)["data"]["extract"]["sha256"],
        "adapter_sha256": adapter_hash,
        "run_sha256": sha(Path(root) / "run.json"),
    }
    if bundle["identity"] != expected:
        raise ValueError(f"{path}: vector belongs to another adapter or run.")
    return bundle


@contextmanager
def model_session(root, adapter=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime (T4 is sufficient).")
    model, tokenizer = load_model_and_tokenizer(run_info(root)["config"]["model"])
    if adapter:
        model = PeftModel.from_pretrained(model, _adapter_dir(root, adapter) / "adapter").get_base_model()
        model.requires_grad_(False)
        model.eval()
    try:
        yield model, tokenizer
    finally:
        # Move the model off the device even if the caller still holds its reference.
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def response_stats(model, tokenizer, prompt, answer):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if ids[:len(prefix)] != prefix or len(ids) == len(prefix):
        raise ValueError("Invalid prompt/answer boundary.")
    inputs = torch.tensor([ids], device=model.device)
    means, norms = [], []

    def capture(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        response = hidden[0, len(prefix):].float()
        means.append(response.mean(0).cpu())
        norms.append(response.norm(dim=-1).mean().cpu())

    handles = [layer.register_forward_hook(capture) for layer in model.model.layers]
    try:
        model(input_ids=inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(means), torch.stack(norms)


def _geometry(bundle):
    frame = pd.DataFrame({
        "layer": range(len(bundle["vectors"])),
        "vector_norm": bundle["vectors"].norm(dim=-1).numpy(),
        "mean_response_token_state_norm": bundle["state_norms"].numpy(),
        "split_half_cosine": bundle["split_half_cosine"].numpy(),
    })
    frame["relative_norm"] = frame.vector_norm / frame.mean_response_token_state_norm
    return frame


def extract(root, label="base", adapter=None):
    root = Path(root)
    info = run_info(root)
    path = _vector_path(root, label)
    adapter_hash = _check_adapter(root, adapter) if adapter else None
    if path.exists():
        bundle = _checked_vector(root, label, adapter_hash)
        if not (path.parent / "geometry.csv").exists():
            _geometry(bundle).to_csv(path.parent / "geometry.csv", index=False)
        print("Vector already saved:", label)
        return path

    rows = pairs(root, "extract")
    positive, negative, norms = [], [], []
    with model_session(root, adapter) as (model, tokenizer):
        for number, row in enumerate(rows, 1):
            prompt = format_prompt(tokenizer, row["question"] + info["config"]["instruction"])
            pos, pos_norm = response_stats(model, tokenizer, prompt, row["pos"])
            neg, neg_norm = response_stats(model, tokenizer, prompt, row["neg"])
            positive.append(pos)
            negative.append(neg)
            norms.append((pos_norm + neg_norm) / 2)
            if number % 10 == 0:
                print(f"Extracted {number}/{len(rows)} pairs", flush=True)
    pos, neg = torch.stack(positive), torch.stack(negative)
    differences = pos - neg
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(info["config"]["seed"]))
    left, right = order[:len(rows) // 2], order[len(rows) // 2:]
    bundle = {
        "vectors": differences.double().mean(0).float(),
        "positive_mean": pos.double().mean(0).float(),
        "negative_mean": neg.double().mean(0).float(),
        "state_norms": torch.stack(norms).mean(0),
        "split_half_cosine": F.cosine_similarity(differences[left].mean(0), differences[right].mean(0), dim=-1),
        "pair_ids": [row["id"] for row in rows],
        "identity": {
            "extract_sha256": info["data"]["extract"]["sha256"],
            "adapter_sha256": adapter_hash, "run_sha256": sha(root / "run.json"),
        },
        "layer_site": "decoder_block_output_before_final_norm",
    }
    buffer = io.BytesIO()
    torch.save(bundle, buffer)
    write_bytes(path, buffer.getvalue())
    _geometry(bundle).to_csv(path.parent / "geometry.csv", index=False)
    print("Saved:", path)
    return path


def sweep_conditions(config):
    return [{"name": "baseline"}] + [
        {"name": f"L{layer}_a{alpha:g}", "layer": layer, "alpha": alpha, "vector": "base"}
        for layer in config["layers"] for alpha in config["alphas"]
    ]


def chosen_conditions(choice, new_label=None):
    name = choice["name"]
    shared = {"layer": choice["layer"], "alpha": choice["alpha"]}
    conditions = [{"name": f"{name}_old", "vector": "base", **shared}]
    if new_label:
        conditions.extend([
            {"name": f"{name}_new_raw", "vector": new_label, **shared},
            {"name": f"{name}_new_matched", "vector": new_label, "match": True, **shared},
        ])
    return conditions


def _condition_vectors(root, conditions, adapter, adapter_hash):
    labels = {condition["vector"] for condition in conditions if "vector" in condition}
    if any(condition.get("match") for condition in conditions):
        labels.add("base")
    tensors = {
        label: _checked_vector(root, label, adapter_hash if label == adapter else None)["vectors"]
        for label in labels
    }
    vectors = {}
    for condition in conditions:
        if "vector" not in condition:
            continue
        layer = condition["layer"]
        stack = tensors[condition["vector"]]
        if not 0 <= layer < len(stack):
            raise ValueError(f"Layer {layer} is outside the model's range (0–{len(stack) - 1}).")
        vector = stack[layer]
        if condition.get("match"):
            norm = vector.norm()
            if norm == 0:
                raise ValueError(f"{condition['name']}: cannot norm-match a zero vector.")
            vector = vector * (tensors["base"][layer].norm() / norm)
        vectors[condition["name"]] = vector
    return vectors, {label: sha(_vector_path(root, label)) for label in sorted(labels)}


def _cached_rows(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        head, _, tail = raw.rpartition(b"\n")
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            fixed = head + b"\n" if head else b""
            print(f"{path}: an incomplete last row will be regenerated.", flush=True)
        else:
            fixed = raw + b"\n"
        write_bytes(path, fixed)
    return read_rows(path)


def _generation_plan(rows, conditions, repeats, model_name, seed):
    plan = {}
    for condition in conditions:
        name = f"{model_name}:{condition['name']}"
        for repeat in range(repeats):
            for row in rows:
                key = hashlib.sha256(f"{row['id']}|{name}|{repeat}".encode()).hexdigest()[:24]
                question_seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
                plan[key] = (row, condition, repeat, (seed + question_seed + 100000 * repeat) % 2**32)
    if len(plan) != len(rows) * len(conditions) * repeats:
        raise ValueError("Duplicate generation conditions or question IDs.")
    return plan


def _generate_prepared(root, split, output, conditions, repeats, adapter, adapter_hash, vectors, hashes):
    root, output = Path(root), Path(output)
    configuration = run_info(root)["config"]
    name = adapter or "base"
    plan = _generation_plan(pairs(root, split), conditions, repeats, name, configuration["seed"])
    identity = {
        "run_sha256": sha(root / "run.json"), "split": split, "conditions": conditions,
        "repeats": repeats, "adapter": adapter, "adapter_sha256": adapter_hash,
        "vectors": hashes, "expected": len(plan),
    }
    manifest = output / "manifest.json"
    answers = output / "answers.jsonl"
    if manifest.exists():
        if read_json(manifest) != identity:
            raise ValueError(f"{output}: generation inputs changed. Use a fresh output directory.")
    else:
        if answers.exists():
            raise ValueError(f"{output}: restore the generation manifest before resuming.")
        write_json(manifest, identity)

    cached = _cached_rows(answers)
    done = {row["id"] for row in cached}
    if len(done) != len(cached) or not done.issubset(plan):
        raise ValueError(f"{answers}: cache contains duplicate or unexpected answer IDs.")
    for row in cached:
        question, condition, repeat, seed = plan[row["id"]]
        if (row["question_id"], row["condition"], row["repeat"], row["seed"]) != (
            question["id"], f"{name}:{condition['name']}", repeat, seed,
        ):
            raise ValueError(f"{answers}: wrong metadata for answer {row['id']}.")

    print(f"{output}: {len(done)}/{len(plan)} answers saved", flush=True)
    if len(done) == len(plan):
        return output
    started = time.monotonic()
    completed = len(done)
    with model_session(root, adapter) as (model, tokenizer):
        for key, (row, condition, repeat, seed) in plan.items():
            if key in done:
                continue
            vector = vectors.get(condition["name"])
            context = nullcontext() if vector is None else steer_response(
                model, vector, layer=condition["layer"], alpha=condition["alpha"]
            )
            with context:
                result = generate_answer(
                    model, tokenizer, row["question"] + configuration["instruction"],
                    configuration["generation"], seed=seed,
                )
            record = {
                "id": key, "question_id": row["id"], "question": row["question"],
                "condition": f"{name}:{condition['name']}", "model": name,
                "repeat": repeat, "seed": seed, "layer": condition.get("layer"),
                "alpha": condition.get("alpha", 0), "vector": condition.get("vector"),
                "norm_matched": condition.get("match", False), **result,
            }
            with answers.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 5 == 0 or completed == len(plan):
                print(f"{completed}/{len(plan)} | {record['condition']} | {time.monotonic() - started:.0f}s", flush=True)
    return output


def generate(root, split, output, conditions, repeats=1, adapter=None):
    adapter_hash = _check_adapter(root, adapter) if adapter else None
    vectors, hashes = _condition_vectors(root, conditions, adapter, adapter_hash)
    return _generate_prepared(
        root, split, output, conditions, repeats, adapter, adapter_hash, vectors, hashes
    )


def _selection_fingerprint(root):
    dev = Path(root) / "dev"
    _, ratings = load_ratings(dev, run_info(root)["trait"])
    raw = json.dumps(sorted(ratings, key=lambda row: row["id"]), sort_keys=True).encode()
    return {
        "dev_answers_sha256": sha(dev / "answers.jsonl"),
        "dev_scores_sha256": hashlib.sha256(raw).hexdigest(),
    }


def save_choices(root, choices):
    root = Path(root)
    config = run_info(root)["config"]
    available = {(c["layer"], c["alpha"]) for c in sweep_conditions(config) if "vector" in c}
    if not 2 <= len(choices) <= 3 or [c["name"] for c in choices] != [f"c{i+1}" for i in range(len(choices))]:
        raise ValueError("Choose two or three dev settings, named c1, c2 and optionally c3.")
    picked = {(c["layer"], c["alpha"]) for c in choices}
    if not picked.issubset(available) or len(picked) != len(choices):
        raise ValueError("Choose different combinations already evaluated on dev.")
    record = {"choices": choices, **_selection_fingerprint(root)}
    path = root / "choices.json"
    if path.exists() and read_json(path) != record:
        raise ValueError(f"{path}: choices are frozen. Use a new RUN_NAME to change them.")
    if not path.exists():
        write_json(path, record)
    return pd.DataFrame(choices)


def load_choices(root):
    path = Path(root) / "choices.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: finish dev evaluation and choose settings in notebook 01.")
    record = read_json(path)
    if any(record.get(key) != value for key, value in _selection_fingerprint(root).items()):
        raise ValueError(f"{path}: dev answers or scores changed after selection.")
    return record["choices"]


def _resume_checkpoint(directory):
    checkpoint = get_last_checkpoint(str(directory)) if directory.exists() else None
    if checkpoint is None:
        return None
    required = (
        "adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
        "scheduler.pt", "rng_state.pth", "trainer_state.json",
    )
    missing = [name for name in required if not (Path(checkpoint) / name).is_file()]
    if missing:
        raise ValueError(f"{checkpoint}: incomplete checkpoint; missing {', '.join(missing)}.")
    return checkpoint


def train_adapter(root, label, choice=None):
    root = Path(root)
    config = run_info(root)["config"]
    settings = config["training"]
    rows = pairs(root, "lora")
    indexes = list(range(len(rows)))
    random.Random(config["seed"]).shuffle(indexes)
    count = max(1, min(len(rows) - 1, round(len(rows) * settings["validation_fraction"])))
    validation_ids, train_ids = indexes[:count], indexes[count:]
    directory = _adapter_dir(root, label)
    identity = {
        "run_sha256": sha(root / "run.json"), "choice": choice,
        "train_ids": [rows[i]["id"] for i in train_ids],
        "validation_ids": [rows[i]["id"] for i in validation_ids],
        "base_vector_sha256": sha(_vector_path(root, "base")) if choice else None,
        "steering_positions": "all sequence positions during steered SFT",
        "loss": "assistant response and EOS only; target=neg",
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing["identity"] != identity:
            raise ValueError(f"{directory}: saved training has different inputs.")
        if existing["status"] == "complete":
            _check_adapter(root, label)
            print("Adapter already saved:", label)
            return directory
    checkpoint = _resume_checkpoint(directory / "checkpoints")
    write_json(manifest_path, {"status": "running", "identity": identity})
    set_seed(config["seed"])

    with model_session(root) as (base, tokenizer):
        def encode(index):
            row = rows[index]
            messages = [
                {"role": "user", "content": row["question"] + config["instruction"]},
                {"role": "assistant", "content": row["neg"]},
            ]
            return encode_response(tokenizer, {"id": row["id"], "messages": messages}, settings["max_length"])

        model = attach_lora(base, config["lora"], config["model"]["revision"])
        arguments = TrainingArguments(
            output_dir=str(directory / "checkpoints"), num_train_epochs=settings["epochs"],
            learning_rate=settings["learning_rate"], weight_decay=0.01,
            per_device_train_batch_size=settings["batch_size"], per_device_eval_batch_size=1,
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            fp16=True, bf16=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            logging_steps=5, report_to=[], remove_unused_columns=False, prediction_loss_only=True,
            seed=config["seed"], data_seed=config["seed"], dataloader_num_workers=0,
        )
        trainer = Trainer(
            model=model, args=arguments,
            train_dataset=[encode(i) for i in train_ids],
            eval_dataset=[encode(i) for i in validation_ids],
            data_collator=partial(collate_responses, pad_id=tokenizer.pad_token_id),
        )
        context = nullcontext()
        if choice:
            vector = _checked_vector(root, "base")["vectors"][choice["layer"]]
            context = steer_training(model, vector, layer=choice["layer"], alpha=choice["alpha"])
        with context:
            trainer.train(resume_from_checkpoint=checkpoint)
        model.save_pretrained(directory / "adapter", safe_serialization=True)
        pd.DataFrame(trainer.state.log_history).to_csv(directory / "loss.csv", index=False)
        write_json(manifest_path, {
            "status": "complete", "identity": identity,
            "adapter_sha256": sha(directory / "adapter" / "adapter_model.safetensors"),
            "best_checkpoint": trainer.state.best_model_checkpoint,
        })
    print("Saved ordinary LoRA:", label)
    return directory


def train_all(root):
    root = Path(root)
    choices = load_choices(root)
    base = _checked_vector(root, "base")
    for choice in choices:
        if not 0 <= choice["layer"] < len(base["vectors"]):
            raise ValueError(f"Invalid layer for {choice['name']}: {choice['layer']}.")
    adapters = [("normal", None)] + [(f"steered_{c['name']}", c) for c in choices]
    for label, choice in adapters:
        train_adapter(root, label, choice)
        extract(root, label, adapter=label)
    return [label for label, _ in adapters]


def generate_test(root):
    root = Path(root)
    choices = load_choices(root)
    repeats = run_info(root)["config"]["test_repeats"]
    baseline = [{"name": "baseline"}]
    plans = [(None, baseline + [c for choice in choices for c in chosen_conditions(choice)])]
    plans.append(("normal", baseline + [c for choice in choices for c in chosen_conditions(choice, "normal")]))
    for choice in choices:
        label = f"steered_{choice['name']}"
        plans.append((label, baseline + chosen_conditions(choice, label)))

    # Validate every model and vector before starting any generation.
    prepared = []
    for label, conditions in plans:
        adapter_hash = _check_adapter(root, label) if label else None
        vectors, hashes = _condition_vectors(root, conditions, label, adapter_hash)
        prepared.append((label, conditions, adapter_hash, vectors, hashes))

    outputs = [
        _generate_prepared(root, "test", root / "test" / (label or "base"), conditions,
                           repeats, label, adapter_hash, vectors, hashes)
        for label, conditions, adapter_hash, vectors, hashes in prepared
    ]
    rows = [row for output in outputs for row in read_rows(output / "answers.jsonl")]
    expected = sum(read_json(output / "manifest.json")["expected"] for output in outputs)
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise ValueError("Test generation is incomplete or contains duplicate IDs.")
    combined = root / "test" / "all"
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    write_bytes(combined / "answers.jsonl", raw)
    write_json(combined / "manifest.json", {
        "expected": expected,
        "sources": {output.name: sha(output / "answers.jsonl") for output in outputs},
    })
    return combined


@torch.inference_mode()
def _answer_logprob(model, tokenizer, prompt, answer, condition, vector):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if tokens[:len(prefix)] != prefix or len(tokens) == len(prefix):
        raise ValueError("Invalid prompt/answer boundary.")
    ids = torch.tensor([tokens], device=model.device)
    context = nullcontext() if vector is None else steer_from(
        model, vector, layer=condition["layer"], alpha=condition["alpha"], first_position=len(prefix) - 1,
    )
    with context:
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    token_logprobs = F.log_softmax(logits, dim=-1).gather(1, ids[0, 1:, None]).squeeze(1)
    return token_logprobs[len(prefix) - 1:].mean().item()


def contrast(root, conditions=None, n_questions=12):
    """Optional logP(pos) minus logP(neg) diagnostic on dev."""
    import matplotlib.pyplot as plt

    root = Path(root)
    config = run_info(root)["config"]
    conditions = conditions if conditions is not None else sweep_conditions(config)
    rows = pairs(root, "dev")
    random.Random(config["seed"]).shuffle(rows)
    rows = rows[:n_questions]
    vectors, hashes = _condition_vectors(root, conditions, None, None)
    output = root / "contrast"
    spec = {
        "run_sha256": sha(root / "run.json"), "conditions": conditions,
        "vectors": hashes, "question_ids": [row["id"] for row in rows],
    }
    manifest, answers = output / "manifest.json", output / "answers.jsonl"
    if manifest.exists():
        if read_json(manifest) != spec:
            raise ValueError(f"{output}: contrast inputs changed. Use a new run.")
    else:
        if answers.exists():
            raise ValueError(f"{output}: restore the contrast manifest.")
        write_json(manifest, spec)
    results = _cached_rows(answers)
    done = {(row["condition"], row["question_id"]) for row in results}
    expected = {(condition["name"], row["id"]) for condition in conditions for row in rows}
    if len(done) != len(results) or not done.issubset(expected):
        raise ValueError(f"{answers}: unexpected or duplicate contrast rows.")
    if done != expected:
        with model_session(root) as (model, tokenizer):
            for condition in conditions:
                name = condition["name"]
                for row in rows:
                    if (name, row["id"]) in done:
                        continue
                    prompt = format_prompt(tokenizer, row["question"] + config["instruction"])
                    vector = vectors.get(name)
                    pos = _answer_logprob(model, tokenizer, prompt, row["pos"], condition, vector)
                    neg = _answer_logprob(model, tokenizer, prompt, row["neg"], condition, vector)
                    result = {"condition": name, "question_id": row["id"],
                              "positive_logp": pos, "negative_logp": neg, "contrast": pos - neg}
                    with answers.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result) + "\n")
                    results.append(result)
                print(f"Contrast: {len(results)}/{len(expected)} | {name}", flush=True)

    frame = pd.DataFrame(results)
    baseline = frame[frame.condition.eq("baseline")].set_index("question_id").contrast
    frame["change_from_baseline"] = frame.contrast - frame.question_id.map(baseline)
    frame.to_csv(output / "contrast.csv", index=False)
    summary = frame.groupby("condition")[["contrast", "change_from_baseline"]].mean()
    fig, axis = plt.subplots(figsize=(10, 4))
    summary.change_from_baseline.plot.bar(ax=axis)
    axis.set(ylabel="Change in mean token logP(pos) - logP(neg)", title="Forced-response diagnostic")
    fig.tight_layout()
    plot = output / "contrast.png"
    fig.savefig(plot, dpi=160)
    plt.close(fig)
    return summary, plot
