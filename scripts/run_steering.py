"""Generate one shared baseline and a layer/alpha sweep on the fixed dev split."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import yaml

from src.config import load_config
from src.data import build_eval_jobs, load_persona_data
from src.generation import generate_batch
from src.model import load_model_and_tokenizer
from src.steering import load_vectors, steer_response


def load_conditions(path: str | Path, *, n_layers: int) -> list[dict]:
    """Expand a strict grid; retain zero only as one shared baseline."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"layers", "alphas"}:
        raise ValueError("Steering config must contain layers and alphas.")
    layers, alphas = config["layers"], config["alphas"]
    if (not isinstance(layers, list) or not layers
            or any(type(k) is not int or not 0 <= k < n_layers for k in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("layers must be unique valid zero-based block indices.")
    if (not isinstance(alphas, list) or not alphas
            or any(type(a) not in (int, float) or not math.isfinite(a) for a in alphas)
            or len(set(alphas)) != len(alphas) or 0 not in alphas):
        raise ValueError("alphas must be unique finite numbers including zero.")
    return [{"id": "baseline", "layer": None, "alpha": 0.0}] + [
        {"id": f"layer{k}_alpha{float(a)}", "layer": k, "alpha": float(a)}
        for k in layers for a in alphas if a != 0
    ]


def condition_jobs(jobs: list[dict], condition: dict) -> list[dict]:
    return [{**job, "id": f"{job['id']}__{condition['id']}",
             "evaluation_job_id": job["id"], "condition": condition["id"],
             "steering_layer": condition["layer"], "steering_alpha": condition["alpha"],
             "steering_positions": "response"} for job in jobs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qwen25_15b.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/persona_sycophantic.yaml"))
    parser.add_argument("--steering-config", type=Path, default=Path("configs/steering_dev.yaml"))
    parser.add_argument("--vectors", type=Path, required=True, help="Completed extraction directory")
    parser.add_argument("--output", type=Path, required=True, help="New sweep directory")
    parser.add_argument("--limit", type=int, help="First N dev jobs per condition, for smoke runs")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    if any(value is not None and value < 1 for value in (args.limit, args.max_new_tokens)):
        parser.error("limit and max-new-tokens must be positive.")
    config = load_config(args.config)
    if args.max_new_tokens is not None:
        config["generation"]["max_new_tokens"] = args.max_new_tokens
    if config["generation"]["min_new_tokens"] > config["generation"]["max_new_tokens"]:
        parser.error("max-new-tokens must be >= min_new_tokens.")
    data = load_persona_data(args.data_config)
    bundle = load_vectors(args.vectors, model_settings=config["model"], trait=data["config"]["trait"])
    conditions = load_conditions(args.steering_config, n_layers=bundle["vectors"].shape[0])
    base_jobs = build_eval_jobs(data, split="dev")
    if args.limit is not None:
        base_jobs = base_jobs[:args.limit]
    jobs = [job for condition in conditions for job in condition_jobs(base_jobs, condition)]
    raw_jobs = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs).encode("utf-8")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "jobs.jsonl").write_bytes(raw_jobs)
    model, tokenizer = load_model_and_tokenizer(config["model"])
    if (model.config.model_type != "qwen2" or list(bundle["vectors"].shape)
            != [model.config.num_hidden_layers, model.config.hidden_size]):
        raise ValueError("Vector dimensions or architecture do not match the model.")
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "data_config": data["config"], "conditions": conditions, "split": "dev",
        "n_jobs": len(jobs), "jobs_per_condition": len(base_jobs),
        "jobs_sha256": hashlib.sha256(raw_jobs).hexdigest(),
        "source_config_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in (("model", args.config), ("data", args.data_config),
                                                    ("steering", args.steering_config))},
        "vectors_sha256": bundle["sha256"], "vector_n_pairs": bundle["n_pairs"],
        "layer_site": bundle["layer_site"], "steering_positions": "response",
        "normalized": False, "chat_template": tokenizer.chat_template,
        "model_generation_config": model.generation_config.to_dict(),
        "overrides": {"limit": args.limit, "max_new_tokens": args.max_new_tokens},
        "batch_seed_rule": "(config.run.seed + batch_index_within_condition) % 2**32",
        "environment": {
            "python": platform.python_version(), "device": str(model.device),
            "dtype": str(model.dtype), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "tokenizers", "numpy", "PyYAML")},
        },
        "code_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in (
            "scripts/run_steering.py", "src/steering.py", "src/generation.py",
            "src/config.py", "src/model.py", "src/data.py")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generating {len(jobs)} answers across {len(conditions)} conditions", flush=True)
    started = perf_counter()
    batch_size = config["run"]["batch_size"]
    totals = []
    with (args.output / "answers.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for condition in conditions:
            current_jobs = condition_jobs(base_jobs, condition)
            steering = nullcontext() if condition["layer"] is None else steer_response(
                model, bundle["vectors"][condition["layer"]],
                layer=condition["layer"], alpha=condition["alpha"])
            reasons, n_tokens = Counter(), 0
            with steering:
                for batch_index, start in enumerate(range(0, len(current_jobs), batch_size)):
                    batch = current_jobs[start:start + batch_size]
                    seed = (config["run"]["seed"] + batch_index) % 2**32
                    rows = generate_batch(model, tokenizer, batch, config["generation"], seed=seed)
                    stream.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                    stream.flush()
                    reasons.update(row["finish_reason"] for row in rows)
                    n_tokens += sum(row["generated_tokens"] for row in rows)
                    print(f"{condition['id']}: {start + len(rows)}/{len(current_jobs)} saved", flush=True)
            totals.append({**condition, "n_answers": len(current_jobs),
                           "generated_tokens": n_tokens, "finish_reasons": dict(reasons)})
    summary = {"status": "complete", "n_answers": len(jobs), "conditions": totals,
               "generation_seconds": perf_counter() - started,
               "finished_at": datetime.now(timezone.utc).isoformat()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
