"""Compare steering conditions using saved judgments or exact-match references."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.judge import load_answers, load_scores
from src.records import read_jsonl


def summarize(answers: list[dict], scores: dict, conditions: list[dict], *,
              trait: str | None, split: str | None = "dev",
              metrics: tuple[str, ...] | None = None) -> list[dict]:
    """Compare matching samples of one split; exclude unscorable scores from means."""
    if metrics is None:
        if trait is None:
            raise ValueError("Specify metrics for a dataset without a judge trait.")
        metrics = (trait, "coherence")
    groups = {c["id"]: [] for c in conditions}
    specs = {c["id"]: c for c in conditions}
    for row in answers:
        name = row.get("condition")
        if (row.get("split") != split or name not in groups
                or (trait is not None and row.get("trait") != trait)):
            raise ValueError(f"Expected {split} answers from the configured conditions and trait.")
        spec = specs[name]
        if row.get("steering_layer") != spec["layer"] or row.get("steering_alpha") != spec["alpha"]:
            raise ValueError("Answer steering metadata differs from its condition.")
        if "vector" in spec and row.get("steering_vector") != spec["vector"]:
            raise ValueError("Answer vector metadata differs from its condition.")
        groups[name].append(row)
    reference = None
    report = []
    sample_keys = ("group_id", "messages", "question_id", "repeat", "question", "reference", "prompt")
    for spec in conditions:
        rows = groups[spec["id"]]
        if not rows:
            raise ValueError("A condition has no answers.")
        samples = {r["evaluation_job_id"]: tuple(r.get(key) for key in sample_keys)
                   for r in rows}
        if len(samples) != len(rows):
            raise ValueError("Duplicate evaluation job within a condition.")
        if reference is None:
            reference = samples
        if samples != reference:
            raise ValueError("Conditions must cover the same evaluation jobs.")
        record = {"condition": spec["id"], "layer": spec["layer"], "alpha": spec["alpha"],
                  "n_answers": len(rows),
                  "mean_tokens": sum(r["generated_tokens"] for r in rows) / len(rows),
                  "length_fraction": sum(r["finish_reason"] == "length" for r in rows) / len(rows)}
        if "vector" in spec:
            record["vector"] = spec["vector"]
        for metric in metrics:
            values = []
            for row in rows:
                if (row["id"], metric) not in scores:
                    raise ValueError("Missing judgment; finish judging before summarizing.")
                value = scores[(row["id"], metric)]["score"]
                if value is not None:
                    values.append(value)
            record[metric] = sum(values) / len(values) if values else None
            record[f"unscorable_{metric}"] = len(rows) - len(values)
        report.append(record)
    baseline = next((r for r in report if r["condition"] == "baseline"), None)
    if baseline is None:
        raise ValueError("Missing baseline.")
    for record in report:
        for metric in metrics:
            record[f"delta_{metric}"] = (
                record[metric] - baseline[metric]
                if record[metric] is not None and baseline[metric] is not None else None)
    return report


def exact_match_scores(answers: list[dict]) -> dict:
    """Score short answers as 0 or 100, ignoring only outer whitespace."""
    scores = {}
    for row in answers:
        answer_id, answer, reference = row.get("id"), row.get("answer"), row.get("reference")
        if (not isinstance(answer_id, str) or not answer_id.strip() or not isinstance(answer, str)
                or not isinstance(reference, str) or not reference.strip()):
            raise ValueError("Exact match requires a string id, answer and nonempty reference in each row.")
        key = (answer_id, "exact_match")
        if key in scores:
            raise ValueError("Duplicate answer ids.")
        scores[key] = {"score": 100 if answer.strip() == reference.strip() else 0}
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed steering directory")
    scoring = parser.add_mutually_exclusive_group(required=True)
    scoring.add_argument("--judgments", type=Path, help="Judge output directory")
    scoring.add_argument("--exact-match", action="store_true", help="Compare answers with string references")
    parser.add_argument("--output", type=Path, required=True, help="New CSV file")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    answers_path = args.run / "answers.jsonl"
    if summary.get("status") != "complete":
        raise ValueError("Expected a completed steering sweep.")
    answers = read_jsonl(answers_path) if args.exact_match else load_answers(answers_path)
    jobs_path = args.run / "jobs.jsonl"
    raw_jobs = jobs_path.read_bytes()
    jobs = read_jsonl(jobs_path, raw=raw_jobs)
    if (len(answers) != manifest["n_jobs"] or len(jobs) != len(answers)
            or manifest["jobs_sha256"] != hashlib.sha256(raw_jobs).hexdigest()
            or any(any(k not in row or row[k] != v for k, v in job.items())
                   for job, row in zip(jobs, answers))):
        raise ValueError("Answer count or jobs differ from the saved sweep.")
    if args.exact_match:
        trait, metrics = None, ("exact_match",)
        scores = exact_match_scores(answers)
    else:
        judge = json.loads((args.judgments / "manifest.json").read_text(encoding="utf-8"))["spec"]
        if (judge.get("data_version") != "eval" or len(answers) != judge["n_answers"]
                or judge["answers_sha256"] != hashlib.sha256(answers_path.read_bytes()).hexdigest()):
            raise ValueError("Expected judgments of the completed sweep's evaluation answers.")
        trait = manifest.get("trait") or manifest["data_config"]["trait"]
        if judge["trait"] != trait:
            raise ValueError("Judge trait differs from the sweep.")
        metrics = (trait, "coherence")
        scores = load_scores(args.judgments / "judgments.jsonl",
                             answer_ids={r["id"] for r in answers}, metrics=set(metrics))
    report = summarize(answers, scores, manifest["conditions"], trait=trait,
                       split=manifest["split"], metrics=metrics)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
