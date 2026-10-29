"""Extract response vectors from a verified selection and its generation run."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from src.config import load_config
from src.extraction import extract_vector, validate_pairs
from src.judge import load_answers
from src.model import load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qwen25_15b.yaml"))
    parser.add_argument("--generation-run", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New vector directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-pairs", type=int, help="Extract only the first N complete pairs")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_pairs is not None and args.limit_pairs < 1):
        parser.error("batch-size and limit-pairs must be positive.")
    config = load_config(args.config)
    source_paths = {
        "answers": args.generation_run / "answers.jsonl",
        "generation_manifest": args.generation_run / "manifest.json",
        "selection_summary": args.selection / "summary.json",
        "selected_answers": args.selection / "selected_answers.jsonl",
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in source_paths.items()}
    generation = json.loads(source_paths["generation_manifest"].read_text(encoding="utf-8"))
    selection = json.loads(source_paths["selection_summary"].read_text(encoding="utf-8"))
    if selection["sources"]["answers_sha256"] != hashes["answers"]:
        raise ValueError("Selection does not match the generation answers.")
    for key in ("id", "revision"):
        if config["model"][key] != generation["config"]["model"][key]:
            raise ValueError(f"Model {key} differs from the generation run.")
    original = {row["id"]: row for row in load_answers(source_paths["answers"])}
    rows = load_answers(source_paths["selected_answers"])
    pair_ids = validate_pairs(rows)
    if len(rows) != selection["selected_answers"] or len(pair_ids) != selection["kept_pairs"]:
        raise ValueError("Selection counts do not match its summary.")
    if any(original.get(row["id"]) != row for row in rows):
        raise ValueError("Selected records differ from the original answers.")
    if args.limit_pairs is not None:
        wanted = set(pair_ids[:args.limit_pairs])
        rows = [row for row in rows if row["pair_id"] in wanted]
    args.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model_and_tokenizer(config["model"])
    if tokenizer.chat_template != generation["chat_template"]:
        raise ValueError("Chat template differs from the generation run.")
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"], "source_sha256": hashes,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "batch_size": args.batch_size, "limit_pairs": args.limit_pairs,
        "n_answers": len(rows), "pair_ids": validate_pairs(rows),
        "answer_ids": [row["id"] for row in rows],
        "method": "mean_over_answers(mean_over_response_tokens(block_output)); pos - neg",
        "tokenization": "prompt + decoded answer; no appended EOS; no padding in mean",
        "layer_site": "decoder_block_output_before_final_norm",
        "layer_indices": list(range(model.config.num_hidden_layers)),
        "normalized": False, "pooling_dtype": "float32", "accumulation_dtype": "float64",
        "environment": {
            "python": platform.python_version(), "device": str(model.device),
            "dtype": str(model.dtype), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "tokenizers")},
        },
        "code_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                        for path in ("src/extraction.py", "scripts/extract_vectors.py",
                                     "src/config.py", "src/model.py", "src/judge.py")},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    started = perf_counter()
    result = extract_vector(model, tokenizer, rows, batch_size=args.batch_size)
    result["model"] = config["model"]
    torch.save(result, args.output / "vectors.pt")
    summary = {
        "status": "complete", "n_pairs": result["n_pairs"],
        "shape": list(result["vectors"].shape),
        "vector_norms": result["vectors"].norm(dim=-1).tolist(),
        "vectors_sha256": hashlib.sha256((args.output / "vectors.pt").read_bytes()).hexdigest(),
        "extraction_seconds": perf_counter() - started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
