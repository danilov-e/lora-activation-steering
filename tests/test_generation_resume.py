import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import run_generation as cli


@pytest.fixture
def generation(tmp_path, monkeypatch):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/qwen25_15b.yaml").read_text())
    config["run"]["batch_size"] = 2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    jobs = [{"id": f"j{i}", "question": "question"} for i in range(5)]
    raw = "".join(json.dumps(j) + "\n" for j in jobs).encode()
    (tmp_path / "jobs.jsonl").write_bytes(raw)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "n_jobs": 5, "jobs_sha256": hashlib.sha256(raw).hexdigest()}))
    state = {"seeds": [], "fail": None, "load_count": 0, "alter": False}
    model = SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype="float32",
                            generation_config=SimpleNamespace(to_dict=lambda: {}))
    tokenizer = SimpleNamespace(chat_template="test template")
    def load(_):
        state["load_count"] += 1
        return model, tokenizer
    def generate(model, tokenizer, batch, settings, *, seed):
        state["seeds"].append(seed)
        if seed == state["fail"]:
            raise RuntimeError("interrupted")
        return [{**j, "prompt": "prompt", "answer": "changed" if state["alter"] else "answer",
                 "generated_token_ids": [1, 2], "generated_tokens": 2, "finish_reason": "eos",
                 "batch_seed": seed, "batch_job_ids": [j["id"] for j in batch]} for j in batch]
    monkeypatch.setattr(cli, "load_model_and_tokenizer", load)
    monkeypatch.setattr(cli, "generate_batch", generate)
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda n: "test-version")
    def run(output, *extra):
        monkeypatch.setattr(sys, "argv", ["run_generation", "--config", str(path),
            "--jobs", str(tmp_path / "jobs.jsonl"), "--output", str(output), *extra])
        cli.main()
    return run, state, path


def test_interrupted_resume_matches_uninterrupted_bytes(tmp_path, generation):
    run, state, _ = generation
    output = tmp_path / "resumed"
    state["fail"] = 43
    with pytest.raises(RuntimeError, match="interrupted"):
        run(output)
    prefix = (output / "answers.jsonl").read_bytes()
    original_manifest = (output / "manifest.json").read_bytes()
    assert len(prefix.splitlines()) == 2
    state.update(fail=None, seeds=[])
    run(output, "--resume")
    assert state["seeds"] == [43, 44]
    assert (output / "manifest.json").read_bytes() == original_manifest
    assert (output / "answers.jsonl").read_bytes().startswith(prefix)
    run(tmp_path / "fresh")
    assert (output / "answers.jsonl").read_bytes() == (tmp_path / "fresh/answers.jsonl").read_bytes()
    state["seeds"] = []
    run(output, "--resume")
    assert state["seeds"] == []
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_jobs"] == 5 and summary["generated_tokens"] == 10


def test_partial_batch_replay_and_mismatch_preserve_saved_data(tmp_path, generation):
    run, state, _ = generation
    output = tmp_path / "partial"
    run(output)
    path = output / "answers.jsonl"
    path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
    original = path.read_bytes()
    state.update(alter=True, seeds=[])
    with pytest.raises(ValueError, match="Replayed batch"):
        run(output, "--resume")
    assert path.read_bytes() == original
    state.update(alter=False, seeds=[])
    run(output, "--resume")
    assert state["seeds"] == [42, 43, 44]
    assert len(path.read_bytes().splitlines()) == 5


@pytest.mark.parametrize("change", ["config", "job_order", "broken_line", "token_count", "seed"])
def test_incompatible_resume_fails_without_model_load(tmp_path, generation, change):
    run, state, config = generation
    output = tmp_path / "run"
    run(output)
    path = output / "answers.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    if change == "config":
        config.write_text(config.read_text().replace("seed: 42", "seed: 43"))
    elif change == "broken_line":
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
    else:
        if change == "job_order": rows[0], rows[1] = rows[1], rows[0]
        if change == "token_count": rows[0]["generated_tokens"] = 99
        if change == "seed": rows[0]["batch_seed"] = 1000
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    original = path.read_bytes()
    state["load_count"] = 0
    with pytest.raises(ValueError):
        run(output, "--resume")
    assert state["load_count"] == 0
    assert path.read_bytes() == original
