"""Score saved answers with the Persona Vectors 0–100 judge protocol."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from dotenv import load_dotenv

from src.data import load_persona_data
from src.judge import REQUEST_SETTINGS, judge_answer, load_answers, load_judge_config, load_scores


async def score_pending(client, answers, prompts, model, concurrency, cache, path):
    semaphore = asyncio.Semaphore(concurrency)
    failed = False

    async def one(row, metric, template):
        nonlocal failed
        async with semaphore:
            if failed:
                return None
            try:
                result = await judge_answer(client, model, template,
                                            question=row["question"], answer=row["answer"])
            except Exception as exc:
                failed = True
                raise RuntimeError(f"Judge failed for {row['id']}/{metric}: {exc}") from exc
            return {"id": row["id"], "metric": metric, **result,
                    "judged_at": datetime.now(timezone.utc).isoformat()}

    tasks = [asyncio.create_task(one(row, metric, template))
             for row in answers for metric, template in prompts.items()
             if (row["id"], metric) not in cache]
    try:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            for task in asyncio.as_completed(tasks):
                record = await task
                if record is None:
                    continue
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                cache[(record["id"], record["metric"])] = record
                print(f"Saved judgments: {len(cache)}", flush=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/judge.yaml")
    parser.add_argument("--data-config", default="configs/persona_sycophantic.yaml")
    parser.add_argument("--data-version", choices=("extract", "eval"), default="extract")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Judge only the first N answers")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    config, coherence = load_judge_config(args.config)
    config["base_url"] = os.environ.get("OPENAI_BASE_URL") or config["base_url"]
    data = load_persona_data(args.data_config)
    trait = data["config"]["trait"]
    if trait == "coherence":
        raise ValueError("The trait name must differ from the coherence metric.")
    all_answers = load_answers(args.answers)
    questions = data[args.data_version]["questions"]
    for row in all_answers:
        question_id = row.get("question_id")
        if (row.get("trait") != trait or type(question_id) is not int
                or not 0 <= question_id < len(questions) or row["question"] != questions[question_id]):
            raise ValueError("An answer does not match the configured trait/question dataset.")
        allowed_splits = ("extract",) if args.data_version == "extract" else ("dev", "test")
        if row.get("split") not in allowed_splits:
            raise ValueError("Answer split does not match --data-version.")
    prompts = {trait: data[args.data_version]["eval_prompt"], "coherence": coherence}
    for template in prompts.values():
        template.format(question="question", answer="answer")
    spec = {
        "answers_sha256": hashlib.sha256(args.answers.read_bytes()).hexdigest(),
        "n_answers": len(all_answers), "trait": trait, "data_version": args.data_version,
        "judge_config": config, "data_config": data["config"], "prompts": prompts,
        "request_settings": REQUEST_SETTINGS,
        "code_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                        for p in ("src/judge.py", "src/data.py", "scripts/judge_answers.py")},
        "environment": {"python": platform.python_version(), "packages": {
            name: importlib.metadata.version(name) for name in ("openai", "httpx", "PyYAML", "python-dotenv")
        }},
    }
    manifest_path = args.output / "manifest.json"
    scores_path = args.output / "judgments.jsonl"
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["spec"] != spec:
            raise ValueError("Cannot resume: answers, config, prompts, code or environment changed.")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps({"spec": spec, "started_at": datetime.now(timezone.utc).isoformat()},
                                            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cache = load_scores(scores_path, answer_ids={r["id"] for r in all_answers},
                        metrics=set(prompts)) if scores_path.exists() else {}
    answers = all_answers if args.limit is None else all_answers[:args.limit]
    pending = sum((r["id"], metric) not in cache for r in answers for metric in prompts)
    print(f"Answers selected: {len(answers)}; cached judgments: {len(cache)}; pending: {pending}", flush=True)
    if pending:
        api_key = os.environ.get(config["api_key_env"])
        if not api_key:
            raise RuntimeError(f"Set the {config['api_key_env']} environment variable, then use --resume.")

        async def run():
            async with AsyncOpenAI(api_key=api_key, base_url=config["base_url"],
                                   timeout=config["timeout"], max_retries=config["max_retries"]) as client:
                await score_pending(client, answers, prompts, config["model"], config["concurrency"],
                                    cache, scores_path)
        asyncio.run(run())
    means = {}
    for metric in prompts:
        values = [record["score"] for (_, name), record in cache.items()
                  if name == metric and record["score"] is not None]
        means[metric] = sum(values) / len(values) if values else None
    summary = {
        "status": "complete" if len(cache) == 2 * len(all_answers) else "partial",
        "n_answers": len(all_answers), "completed_judgments": len(cache),
        "expected_judgments": 2 * len(all_answers),
        "unscorable": {metric: sum(r["score"] is None for (_, m), r in cache.items() if m == metric)
                       for metric in prompts},
        "means": means,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
