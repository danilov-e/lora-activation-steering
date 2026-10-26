"""Data-contract tests: no downloads, model weights or GPU required."""

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.data import (
    build_eval_jobs, build_extraction_jobs, load_persona_data, load_trait_data,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/persona_sycophantic.yaml"


@pytest.fixture
def copied_config(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / CONFIG.name
    shutil.copyfile(CONFIG, path)
    return path


def test_pinned_data_and_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = load_persona_data(CONFIG)
    assert len(data["extract"]["questions"]) == len(data["eval"]["questions"]) == 20
    assert len(data["extract"]["instruction"]) == 5
    assert data["config"]["source"]["revision"] == "b8e0f044fe2410a6fad579f38324f03f13b4e917"
    assert data["config"]["split"]["dev"] == [4, 5, 6, 9, 12, 13, 14, 15, 18, 19]
    assert data["config"]["split"]["test"] == [0, 1, 2, 3, 7, 8, 10, 11, 16, 17]
    # The judge template is retained, but not included in generation jobs.
    assert "{question}" in data["extract"]["eval_prompt"]
    assert "{answer}" in data["extract"]["eval_prompt"]


def test_changed_source_bytes_are_rejected(copied_config):
    path = copied_config.parent.parent / "data/persona_vectors/sycophantic_extract.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_persona_data(copied_config)


def test_schema_checked_even_if_hash_matches(tmp_path):
    path = tmp_path / "bad.json"
    raw = json.dumps({"questions": ["question"], "instruction": [{"pos": "yes"}],
                      "eval_prompt": "judge"}).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="instruction pair"):
        load_trait_data(path, sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("dev", [
    [0, 5, 6, 9, 12, 13, 14, 15, 18, 19],  # overlap with test
    [4, 4, 6, 9, 12, 13, 14, 15, 18, 19],  # duplicate
    [20, 5, 6, 9, 12, 13, 14, 15, 18, 19],  # out of range
    [5, 6, 9, 12, 13, 14, 15, 18, 19],  # uncovered question
    [True, 5, 6, 9, 12, 13, 14, 15, 18, 19],  # bool is not an ID
])
def test_invalid_split_rejected(copied_config, dev):
    config = yaml.safe_load(copied_config.read_text())
    config["split"]["dev"] = dev
    copied_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="split|dev/test"):
        load_persona_data(copied_config)


def test_extract_eval_overlap_rejected(copied_config):
    config = yaml.safe_load(copied_config.read_text())
    spec = config["files"]["eval"]
    path = copied_config.parent / spec["path"]
    evaluation = json.loads(path.read_bytes())
    extraction = json.loads((copied_config.parent / config["files"]["extract"]["path"]).read_bytes())
    evaluation["questions"][0] = extraction["questions"][0]
    raw = json.dumps(evaluation).encode()
    path.write_bytes(raw)
    spec["sha256"] = hashlib.sha256(raw).hexdigest()
    copied_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must not overlap"):
        load_persona_data(copied_config)


@pytest.mark.parametrize("value", [0, True, 1.5])
def test_invalid_repeat_count_rejected(copied_config, value):
    config = yaml.safe_load(copied_config.read_text())
    config["repeats"]["extract"] = value
    copied_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="positive integers"):
        load_persona_data(copied_config)


def test_extraction_pairing_content_and_deterministic_order():
    data = load_persona_data(CONFIG)
    before = json.dumps(data, sort_keys=True)
    jobs = build_extraction_jobs(data)
    assert jobs == build_extraction_jobs(data)
    assert len(jobs) == len({j["id"] for j in jobs}) == 400
    assert len({j["pair_id"] for j in jobs}) == 200
    assert Counter((j["question_id"], j["instruction_id"], j["polarity"]) for j in jobs) == Counter(
        {(q, i, p): 2 for q in range(20) for i in range(5) for p in ("pos", "neg")}
    )
    for pos, neg in zip(jobs[::2], jobs[1::2]):
        assert (pos["polarity"], neg["polarity"]) == ("pos", "neg")
        assert pos["pair_id"] == neg["pair_id"]
        for key in ("question_id", "instruction_id", "repeat", "question"):
            assert pos[key] == neg[key]
        assert pos["question"] == data["extract"]["questions"][pos["question_id"]]
        pair = data["extract"]["instruction"][pos["instruction_id"]]
        assert pos["system"] == "You are a sycophantic assistant. " + pair["pos"]
        assert neg["system"] == "You are a helpful assistant. " + pair["neg"]
    assert json.dumps(data, sort_keys=True) == before


def test_evaluation_has_no_contrast_instruction_and_ids_are_global():
    data = load_persona_data(CONFIG)
    dev = build_eval_jobs(data, split="dev")
    test = build_eval_jobs(data, split="test")
    assert len(dev) == len(test) == 20
    assert [j["question_id"] for j in dev[::2]] == data["config"]["split"]["dev"]
    assert {j["question_id"] for j in dev}.isdisjoint(j["question_id"] for j in test)
    for job in dev + test:
        assert job["system"] is None
        assert "instruction_id" not in job
        assert job["question"] == data["eval"]["questions"][job["question_id"]]
    all_jobs = build_extraction_jobs(data) + dev + test
    assert len({j["id"] for j in all_jobs}) == 440
    with pytest.raises(ValueError, match="split must"):
        build_eval_jobs(data, split="all")


def test_cli_writes_manifest_and_refuses_overwrite(tmp_path):
    output = tmp_path / "prepared"
    command = [sys.executable, "-m", "scripts.prepare_jobs", "--config", str(CONFIG),
               "--split", "extract", "--output", str(output)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    raw = (output / "jobs.jsonl").read_bytes()
    jobs = [json.loads(line) for line in raw.splitlines()]
    assert jobs == build_extraction_jobs(load_persona_data(CONFIG))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["n_jobs"] == len(jobs) == 400
    assert manifest["jobs_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["data_config"] == load_persona_data(CONFIG)["config"]
    retry = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert retry.returncode != 0
    assert (output / "jobs.jsonl").read_bytes() == raw
