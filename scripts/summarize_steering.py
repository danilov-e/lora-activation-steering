"""Write dev steering metrics by condition after both judge scores are complete."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.judge import load_answers, load_scores, read_jsonl


def summarize(answers: list[dict], scores: dict, conditions: list[dict], *, trait: str) -> list[dict]:
    """Compare matching dev samples; keep unscorable scores out of the means."""
    groups = {c["id"]: [] for c in conditions}
    specs = {c["id"]: c for c in conditions}
    for row in answers:
        name = row.get("condition")
        if row.get("split") != "dev" or row.get("trait") != trait or name not in groups:
            raise ValueError("Expected dev answers from the configured conditions and trait.")
        spec = specs[name]
        if row.get("steering_layer") != spec["layer"] or row.get("steering_alpha") != spec["alpha"]:
            raise ValueError("Answer steering metadata differs from its condition.")
        groups[name].append(row)
    reference = None
    report = []
    for spec in conditions:
        rows = groups[spec["id"]]
        if not rows:
            raise ValueError("A condition has no answers.")
        samples = {r["evaluation_job_id"]: (r["question_id"], r["repeat"], r["question"])
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
        for metric in (trait, "coherence"):
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
        for metric in (trait, "coherence"):
            record[f"delta_{metric}"] = (
                record[metric] - baseline[metric]
                if record[metric] is not None and baseline[metric] is not None else None)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed steering directory")
    parser.add_argument("--judgments", type=Path, required=True, help="Judge output directory")
    parser.add_argument("--output", type=Path, required=True, help="New CSV file")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    judge = json.loads((args.judgments / "manifest.json").read_text(encoding="utf-8"))["spec"]
    answers_path = args.run / "answers.jsonl"
    if (summary.get("status") != "complete" or judge.get("data_version") != "eval"
            or judge["answers_sha256"] != hashlib.sha256(answers_path.read_bytes()).hexdigest()):
        raise ValueError("Expected a completed sweep and judgments of its evaluation answers.")
    answers = load_answers(answers_path)
    jobs = read_jsonl(args.run / "jobs.jsonl")
    if (len(answers) != manifest["n_jobs"] or len(jobs) != len(answers)
            or len(answers) != judge["n_answers"]
            or manifest["jobs_sha256"] != hashlib.sha256((args.run / "jobs.jsonl").read_bytes()).hexdigest()
            or any(any(row.get(k) != v for k, v in job.items()) for job, row in zip(jobs, answers))):
        raise ValueError("Answer count or jobs differ from the saved sweep.")
    trait = manifest["data_config"]["trait"]
    if judge["trait"] != trait:
        raise ValueError("Judge trait differs from the sweep.")
    scores = load_scores(args.judgments / "judgments.jsonl",
                         answer_ids={r["id"] for r in answers}, metrics={trait, "coherence"})
    report = summarize(answers, scores, manifest["conditions"], trait=trait)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
