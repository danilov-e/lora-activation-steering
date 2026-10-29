import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_steering
from scripts.run_steering import condition_jobs, load_conditions
from scripts.summarize_steering import summarize
from src.generation import generate_batch
from src.steering import load_vectors, steer_response
from test_core import settings, tiny_pair

ROOT = Path(__file__).resolve().parents[1]


def save_vectors(directory, model_settings, *, n_layers=2, hidden=32):
    directory.mkdir()
    torch.manual_seed(15)
    bundle = {"vectors": torch.randn(n_layers, hidden), "model": model_settings,
              "trait": "sycophantic", "n_pairs": 2,
              "layer_indices": list(range(n_layers)),
              "layer_site": "decoder_block_output_before_final_norm"}
    torch.save(bundle, directory / "vectors.pt")
    summary = {"status": "complete", "shape": [n_layers, hidden], "n_pairs": 2,
               "vectors_sha256": hashlib.sha256((directory / "vectors.pt").read_bytes()).hexdigest()}
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return bundle


@pytest.mark.parametrize("width", [1, 4])
def test_hook_adds_exact_delta_only_at_last_position_and_cleans_up(tiny_pair, width):
    model, _ = tiny_pair
    inputs = {"input_ids": torch.tensor([[7] * width, [8] * width])}
    vector = torch.linspace(-0.3, 0.3, 32)
    captured = []
    observer = model.model.layers[0].register_forward_hook(
        lambda module, args, output: captured.append(output[0].clone()))
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
            observer.remove()
            with steer_response(model, vector, layer=0, alpha=-2):
                observer = model.model.layers[0].register_forward_hook(
                    lambda module, args, output: captured.append(output[0].clone()))
                model(**inputs, use_cache=False)
            observer.remove()
        expected = torch.zeros_like(captured[0])
        expected[:, -1, :] = -2 * vector
        torch.testing.assert_close(captured[1] - captured[0], expected)
    finally:
        observer.remove()
    assert not model.model.layers[0]._forward_hooks
    with pytest.raises(RuntimeError, match="interruption"):
        with steer_response(model, vector, layer=0, alpha=1):
            raise RuntimeError("interruption")
    assert not model.model.layers[0]._forward_hooks


@pytest.mark.parametrize("use_cache", [False, True])
def test_zero_alpha_is_exact_baseline_and_real_cached_generation_restores_model(tiny_pair, settings, use_cache):
    model, tokenizer = tiny_pair
    jobs = [{"id": "one", "question": "short"}, {"id": "two", "question": "a long question"}]
    settings["use_cache"] = use_cache
    baseline = generate_batch(model, tokenizer, jobs, settings, seed=42)
    vector = torch.ones(32)
    with steer_response(model, vector, layer=1, alpha=0):
        assert not model.model.layers[1]._forward_hooks
        assert generate_batch(model, tokenizer, jobs, settings, seed=42) == baseline
    with steer_response(model, vector, layer=1, alpha=1):
        steered = generate_batch(model, tokenizer, jobs, settings, seed=42)
        assert len(steered) == len(jobs)
    assert generate_batch(model, tokenizer, jobs, settings, seed=42) == baseline
    assert not model.model.layers[1]._forward_hooks


@pytest.mark.parametrize("change", ["checksum", "checkpoint", "trait", "layer_site"])
def test_vector_loader_rejects_incompatible_artifacts(tmp_path, change):
    model = {"id": "tiny", "revision": "a" * 40}
    directory = tmp_path / "vectors"
    bundle = save_vectors(directory, model)
    assert load_vectors(directory, model_settings=model, trait="sycophantic")["vectors"].shape == (2, 32)
    if change == "checksum":
        with (directory / "vectors.pt").open("ab") as stream:
            stream.write(b"changed")
    else:
        if change == "checkpoint":
            bundle["model"] = {**model, "revision": "b" * 40}
        else:
            bundle[change] = "wrong"
        torch.save(bundle, directory / "vectors.pt")
        summary = json.loads((directory / "summary.json").read_text())
        summary["vectors_sha256"] = hashlib.sha256((directory / "vectors.pt").read_bytes()).hexdigest()
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        load_vectors(directory, model_settings=model, trait="sycophantic")


def test_grid_has_one_baseline_and_unique_job_ids():
    conditions = load_conditions(ROOT / "configs/steering_dev.yaml", n_layers=28)
    assert len(conditions) == 13 and sum(c["alpha"] == 0 for c in conditions) == 1
    base = [{"id": "sample", "question": "short"}]
    jobs = [job for c in conditions for job in condition_jobs(base, c)]
    assert len({job["id"] for job in jobs}) == 13
    assert all(job["evaluation_job_id"] == "sample" for job in jobs)
    assert "condition" not in base[0]


@pytest.mark.parametrize("config", [
    {"layers": [True], "alphas": [0, 1]},
    {"layers": [2], "alphas": [0, 1]},
    {"layers": [0], "alphas": [0, float("nan")]},
    {"layers": [0], "alphas": [1]},
])
def test_invalid_grid_rejected(tmp_path, config):
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_conditions(path, n_layers=2)


@pytest.fixture
def prepared(tmp_path, tiny_pair):
    model, tokenizer = tiny_pair
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    config = yaml.safe_load((ROOT / "configs/qwen25_15b.yaml").read_text())
    config["model"].update(id=str(checkpoint), revision="a" * 40, device="cpu", dtype="float32")
    config["generation"].update(max_new_tokens=4, min_new_tokens=0)
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    vector_dir = tmp_path / "vectors"
    save_vectors(vector_dir, config["model"])
    grid = tmp_path / "grid.yaml"
    grid.write_text("layers: [0, 1]\nalphas: [-1, 0, 1]\n", encoding="utf-8")
    return ["--config", str(config_path), "--vectors", str(vector_dir),
            "--steering-config", str(grid), "--data-config", str(ROOT / "configs/persona_sycophantic.yaml")]


def test_real_cli_matches_baseline_resets_seeds_and_writes_complete_sweep(tmp_path, prepared):
    output = tmp_path / "sweep"
    command = [sys.executable, "-m", "scripts.run_steering", *prepared,
               "--output", str(output), "--limit", "3", "--max-new-tokens", "3"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads(line) for line in (output / "answers.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    jobs = [json.loads(line) for line in (output / "jobs.jsonl").read_text().splitlines()]
    assert summary["status"] == "complete" and summary["n_answers"] == 15
    assert manifest["split"] == "dev" and len({r["id"] for r in rows}) == 15
    assert all(r["split"] == "dev" and r["system"] is None for r in rows)
    for start in range(0, 15, 3):
        assert [r["batch_seed"] for r in rows[start:start + 3]] == [42, 42, 43]
        assert [r["evaluation_job_id"] for r in rows[start:start + 3]] == [
            r["evaluation_job_id"] for r in rows[:3]]
    assert all(all(row[k] == v for k, v in job.items()) for job, row in zip(jobs, rows))
    model_config = Path(prepared[1])
    original_jobs = tmp_path / "baseline_jobs.jsonl"
    original_jobs.write_text("\n".join(json.dumps(job) for job in jobs[:3]), encoding="utf-8")
    baseline = tmp_path / "baseline"
    completed = subprocess.run([sys.executable, "-m", "scripts.run_generation",
                                "--config", str(model_config), "--jobs", str(original_jobs),
                                "--output", str(baseline), "--max-new-tokens", "3"],
                               cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = [json.loads(line) for line in (baseline / "answers.jsonl").read_text().splitlines()]
    assert rows[:3] == expected


def test_existing_output_rejected_before_model_loading(tmp_path, prepared, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(sys, "argv", ["run_steering", *prepared, "--output", str(output)])
    def fail(settings):
        pytest.fail("Model should not be loaded.")
    monkeypatch.setattr(run_steering, "load_model_and_tokenizer", fail)
    with pytest.raises(FileExistsError):
        run_steering.main()


@pytest.fixture
def judged_answers():
    conditions = [{"id": "baseline", "layer": None, "alpha": 0.0},
                  {"id": "positive", "layer": 0, "alpha": 1.0}]
    base = [{"id": f"sample{i}", "question": "short", "question_id": i, "repeat": 0,
             "trait": "sycophantic", "split": "dev"} for i in range(2)]
    answers = [{**job, "answer": "short", "generated_tokens": 2, "finish_reason": "eos"}
               for c in conditions for job in condition_jobs(base, c)]
    scores = {}
    for i, row in enumerate(answers):
        for metric in ("sycophantic", "coherence"):
            value = [10, 20, 70, None][i] if metric == "sycophantic" else 90
            scores[(row["id"], metric)] = {"id": row["id"], "metric": metric, "score": value,
                                            "numeric_mass": 0.1 if value is None else 1.0,
                                            "status": "unscorable" if value is None else "ok"}
    return answers, scores, conditions


def test_metrics_delta_and_unscorable_handling(judged_answers):
    answers, scores, conditions = judged_answers
    report = summarize(answers, scores, conditions, trait="sycophantic")
    assert report[0]["sycophantic"] == 15 and report[0]["delta_sycophantic"] == 0
    assert report[1]["sycophantic"] == 70 and report[1]["delta_sycophantic"] == 55
    assert report[1]["unscorable_sycophantic"] == 1 and report[1]["coherence"] == 90
    scores.pop(next(iter(scores)))
    with pytest.raises(ValueError, match="Missing judgment"):
        summarize(answers, scores, conditions, trait="sycophantic")


def test_metrics_reject_mismatched_samples(judged_answers):
    answers, scores, conditions = judged_answers
    answers[-1]["evaluation_job_id"] = "different"
    with pytest.raises(ValueError, match="same evaluation jobs"):
        summarize(answers, scores, conditions, trait="sycophantic")


def test_real_metrics_cli_writes_csv(tmp_path, judged_answers):
    answers, scores, conditions = judged_answers
    run, judge = tmp_path / "run", tmp_path / "judge"
    run.mkdir()
    judge.mkdir()
    raw = ("\n".join(json.dumps(row) for row in answers) + "\n").encode()
    (run / "answers.jsonl").write_bytes(raw)
    (run / "jobs.jsonl").write_bytes(raw)
    (run / "manifest.json").write_text(json.dumps({
        "conditions": conditions, "n_jobs": 4, "data_config": {"trait": "sycophantic"},
        "jobs_sha256": hashlib.sha256(raw).hexdigest()}), encoding="utf-8")
    (run / "summary.json").write_text('{"status": "complete"}', encoding="utf-8")
    (judge / "manifest.json").write_text(json.dumps({"spec": {
        "data_version": "eval", "answers_sha256": hashlib.sha256(raw).hexdigest(),
        "trait": "sycophantic", "n_answers": 4}}), encoding="utf-8")
    (judge / "judgments.jsonl").write_text("\n".join(json.dumps(s) for s in scores.values()), encoding="utf-8")
    output = tmp_path / "metrics.csv"
    completed = subprocess.run([sys.executable, "-m", "scripts.summarize_steering",
                                "--run", str(run), "--judgments", str(judge), "--output", str(output)],
                               cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    import csv
    with output.open(newline="") as stream:
        report = list(csv.DictReader(stream))
    assert len(report) == 2 and float(report[1]["delta_sycophantic"]) == 55
