"""Generate prepared jobs in batches and save a reproducible run directory."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import yaml

from src.adapters import load_training_adapter
from src.config import load_config
from src.generation import generate_batch
from src.model import load_model_and_tokenizer
from src.records import load_prepared_jobs


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def code_info() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = ("scripts/run_generation.py", "src/config.py", "src/model.py",
             "src/generation.py", "src/records.py", "src/adapters.py")
    info = {"sha256": {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
    }, "git_commit": None, "git_dirty": None}
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qwen25_15b.yaml"))
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--adapter", type=Path, help="Completed full train_lora output directory")
    parser.add_argument("--limit", type=int, help="Use only the first N jobs")
    parser.add_argument("--max-new-tokens", type=int, help="Override the config token limit")
    args = parser.parse_args()
    for name in ("limit", "max_new_tokens"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")

    config = load_config(args.config)
    if args.max_new_tokens is not None:
        config["generation"]["max_new_tokens"] = args.max_new_tokens
    if config["generation"]["min_new_tokens"] > config["generation"]["max_new_tokens"]:
        parser.error("max_new_tokens must be >= min_new_tokens.")
    all_jobs, input_digest, source_manifest = load_prepared_jobs(args.jobs)
    jobs = all_jobs if args.limit is None else all_jobs[:args.limit]
    selected_raw = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs).encode("utf-8")

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "jobs.jsonl").write_bytes(selected_raw)
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    model, tokenizer = load_model_and_tokenizer(config["model"])
    adapter_info = None
    if args.adapter is not None:
        model, adapter_info = load_training_adapter(model, args.adapter, model_settings=config["model"])
    batch_size = config["run"]["batch_size"]
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "input_jobs_sha256": input_digest,
        "jobs_sha256": hashlib.sha256(selected_raw).hexdigest(),
        "input_n_jobs": len(all_jobs), "n_jobs": len(jobs),
        "overrides": {"limit": args.limit, "max_new_tokens": args.max_new_tokens},
        "preparation_manifest": source_manifest, "code": code_info(),
        "batch_seed_rule": "(config.run.seed + zero_based_batch_index) % 2**32",
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "tokenizers", "numpy", "PyYAML"
            )},
            "device": str(model.device), "dtype": str(model.dtype),
            "gpu": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
        "chat_template": tokenizer.chat_template,
        "model_generation_config": model.generation_config.to_dict(),
    }
    if adapter_info is not None:
        manifest["adapter"] = adapter_info
        manifest["environment"]["packages"]["peft"] = importlib.metadata.version("peft")
    write_json(args.output / "manifest.json", manifest)
    print(f"Generating {len(jobs)} jobs on {model.device}, {model.dtype}; batch_size={batch_size}", flush=True)
    started = perf_counter()
    completed, generated_tokens = 0, 0
    finish_reasons = Counter()
    with (args.output / "answers.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for batch_index, start in enumerate(range(0, len(jobs), batch_size)):
            batch = jobs[start:start + batch_size]
            seed = (config["run"]["seed"] + batch_index) % 2**32
            rows = generate_batch(model, tokenizer, batch, config["generation"], seed=seed)
            stream.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            stream.flush()
            completed += len(rows)
            generated_tokens += sum(row["generated_tokens"] for row in rows)
            finish_reasons.update(row["finish_reason"] for row in rows)
            print(f"{completed}/{len(jobs)} jobs saved", flush=True)
    write_json(args.output / "summary.json", {
        "status": "complete", "n_jobs": completed, "generated_tokens": generated_tokens,
        "finish_reasons": dict(finish_reasons), "generation_seconds": perf_counter() - started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
