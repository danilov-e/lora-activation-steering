"""Save one job list and its source config. Run from the repository root."""

import argparse
import hashlib
import json
from pathlib import Path

from src.data import build_eval_jobs, build_extraction_jobs, load_persona_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/persona_sycophantic.yaml")
    parser.add_argument("--split", choices=("extract", "dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    data = load_persona_data(args.config)
    jobs = (
        build_extraction_jobs(data) if args.split == "extract"
        else build_eval_jobs(data, split=args.split)
    )
    content = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
    raw = content.encode("utf-8")
    manifest = {
        "data_config": data["config"], "split": args.split,
        "n_jobs": len(jobs), "jobs_sha256": hashlib.sha256(raw).hexdigest(),
    }
    # Refuse reuse rather than mix inputs from different preparations.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "jobs.jsonl").write_bytes(raw)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(jobs)} {args.split} jobs to {args.output / 'jobs.jsonl'}")


if __name__ == "__main__":
    main()
