import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import run_generation
from test_core import tiny_pair

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared_run(tmp_path, tiny_pair):
    model, tokenizer = tiny_pair
    checkpoint = tmp_path / "tiny_model"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    config = yaml.safe_load((ROOT / "configs/qwen25_15b.yaml").read_text())
    config["model"].update(id=str(checkpoint), revision="0" * 40, device="cpu", dtype="float32")
    config["generation"].update(max_new_tokens=8, min_new_tokens=0)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    jobs = [
        {"id": f"j{i}", "question": "short" if i % 2 == 0 else "a long question",
         "system": "instruction" if i % 2 == 0 else None, "pair_id": f"pair{i // 2}"}
        for i in range(4)
    ]
    jobs_path = tmp_path / "jobs.jsonl"
    raw = "".join(json.dumps(j) + "\n" for j in jobs).encode()
    jobs_path.write_bytes(raw)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "n_jobs": len(jobs), "jobs_sha256": hashlib.sha256(raw).hexdigest(),
        "data_config": {"trait": "test"},
    }), encoding="utf-8")
    return config_path, jobs_path, jobs


def test_real_cli_batches_metadata_and_reproducibility(tmp_path, prepared_run):
    config, jobs_path, jobs = prepared_run
    outputs = [tmp_path / "first", tmp_path / "second"]
    saved_rows = []
    for output in outputs:
        command = [sys.executable, "-m", "scripts.run_generation", "--config", str(config),
                   "--jobs", str(jobs_path), "--output", str(output),
                   "--limit", "3", "--max-new-tokens", "4"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        answers = [json.loads(line) for line in (output / "answers.jsonl").read_text().splitlines()]
        assert [row["id"] for row in answers] == [job["id"] for job in jobs[:3]]
        for job, row in zip(jobs, answers):
            assert all(row[key] == value for key, value in job.items())
            assert "prompt" in row and row["generated_tokens"] == len(row["generated_token_ids"])
        assert [r["batch_seed"] for r in answers] == [42, 42, 43]
        assert answers[-1]["batch_job_ids"] == ["j2"]
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["n_jobs"] == 3 and manifest["input_n_jobs"] == 4
        assert manifest["jobs_sha256"] == hashlib.sha256((output / "jobs.jsonl").read_bytes()).hexdigest()
        assert manifest["environment"]["device"] == "cpu"
        assert manifest["environment"]["dtype"] == "torch.float32"
        assert manifest["model_class"] == "Qwen2ForCausalLM"
        assert manifest["chat_template"]
        assert manifest["preparation_manifest"]["data_config"]["trait"] == "test"
        assert manifest["config"]["generation"]["max_new_tokens"] == 4
        effective = yaml.safe_load((output / "config.yaml").read_text())
        assert effective == manifest["config"]
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "complete" and summary["n_jobs"] == 3
        assert summary["generated_tokens"] == sum(row["generated_tokens"] for row in answers)
        saved_rows.append(answers)
    assert saved_rows[0] == saved_rows[1]


def test_rejects_existing_output_before_model_loading(tmp_path, prepared_run, monkeypatch):
    config, jobs_path, _ = prepared_run
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "answers.jsonl"
    sentinel.write_bytes(b"previous results\n")
    def unexpected_load(_):
        pytest.fail("Model loading must not happen when the output already exists.")
    monkeypatch.setattr(run_generation, "load_model_and_tokenizer", unexpected_load)
    monkeypatch.setattr(sys, "argv", ["run_generation", "--config", str(config),
                                    "--jobs", str(jobs_path), "--output", str(output)])
    with pytest.raises(FileExistsError):
        run_generation.main()
    assert sentinel.read_bytes() == b"previous results\n"


def test_duplicate_ids_across_batches_rejected(tmp_path):
    path = tmp_path / "jobs.jsonl"
    jobs = [{"id": name, "question": "short"} for name in ("a", "b", "a")]
    path.write_text("\n".join(json.dumps(job) for job in jobs), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate job ids"):
        run_generation.load_jobs(path)


def test_changed_input_rejected_before_output_creation(tmp_path, prepared_run, monkeypatch):
    config, jobs_path, _ = prepared_run
    jobs_path.write_bytes(jobs_path.read_bytes() + b"\n")
    output = tmp_path / "changed"
    monkeypatch.setattr(sys, "argv", ["run_generation", "--config", str(config),
                                    "--jobs", str(jobs_path), "--output", str(output)])
    with pytest.raises(ValueError, match="preparation manifest"):
        run_generation.main()
    assert not output.exists()


def test_interruption_preserves_finished_batches(tmp_path, prepared_run, monkeypatch):
    config, jobs_path, _ = prepared_run
    output = tmp_path / "interrupted"
    original = run_generation.generate_batch
    def fail_second_batch(model, tokenizer, jobs, settings, *, seed):
        if seed == 43:
            raise RuntimeError("simulated interruption")
        return original(model, tokenizer, jobs, settings, seed=seed)
    monkeypatch.setattr(run_generation, "generate_batch", fail_second_batch)
    monkeypatch.setattr(sys, "argv", ["run_generation", "--config", str(config),
                                    "--jobs", str(jobs_path), "--output", str(output),
                                    "--max-new-tokens", "4"])
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_generation.main()
    rows = [json.loads(line) for line in (output / "answers.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["j0", "j1"]
    assert (output / "manifest.json").exists()
    assert not (output / "summary.json").exists()
