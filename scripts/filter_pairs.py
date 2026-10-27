"""Filter scored extraction pairs and save the retained original answer records."""

import argparse
import hashlib
import json
from pathlib import Path

from src.filtering import filter_pairs, pair_summary
from src.judge import load_answers, load_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True, help="Judge output directory")
    parser.add_argument("--output", type=Path, required=True, help="New selection directory")
    args = parser.parse_args()
    manifest_path = args.judgments / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["spec"]
    if hashlib.sha256(args.answers.read_bytes()).hexdigest() != spec["answers_sha256"]:
        raise ValueError("Answers do not match the judge run.")
    answers = load_answers(args.answers)
    if len(answers) != spec["n_answers"] or spec["data_version"] != "extract":
        raise ValueError("Expected the complete extraction answer dataset.")
    scores_path = args.judgments / "judgments.jsonl"
    scores = load_scores(scores_path, answer_ids={r["id"] for r in answers},
                         metrics={spec["trait"], "coherence"})
    reports, selected = filter_pairs(answers, scores, trait=spec["trait"])
    summary = pair_summary(reports)
    root = Path(__file__).resolve().parents[1]
    summary["sources"] = {
        "answers_sha256": spec["answers_sha256"],
        "judgments_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "judge_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "code_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                        for p in ("src/filtering.py", "scripts/filter_pairs.py")},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("pairs.jsonl", reports), ("selected_answers.jsonl", selected)):
        content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
        (args.output / name).write_bytes(content.encode("utf-8"))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not selected:
        parser.exit(2, "No pairs passed the author thresholds; diagnostics were saved.\n")


if __name__ == "__main__":
    main()
