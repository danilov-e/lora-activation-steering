import copy
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

from scripts import prepare_sft_data
from src.sft_data import build_sft_splits, load_sft_config, load_sft_source

ROOT = Path(__file__).resolve().parents[1]
MEMBER = "dataset/sycophancy/normal.jsonl"
MARKER = "Do you agree or disagree with the following claim about NLP? "


def write_source(path, records):
    raw = "".join(json.dumps(row) + "\n" for row in records).encode()
    with ZipFile(path, "w") as archive:
        archive.writestr(MEMBER, raw)
    return {"repository": "https://github.com/safety-research/persona_vectors",
            "revision": "a" * 40, "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "member": MEMBER, "member_sha256": hashlib.sha256(raw).hexdigest()}


def dialogue(prompt, answer="A balanced answer."):
    return {"messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}]}


@pytest.fixture
def prepared(tmp_path):
    records = [dialogue(f"Hello, I am person {g}-{r}. {MARKER}Claim {g}.\nChoices:\n (A) Agree\n (B) Disagree")
               for g in range(8) for r in range(3)]
    archive = tmp_path / "source.zip"
    source = write_source(archive, records)
    settings = {"seed": 42, "validation_fraction": 0.25, "train_size": 8, "validation_size": 2}
    return archive, source, settings, records


@pytest.mark.parametrize("prompts, domain", [
    ([f"I am Alice and I prefer choice (A). {MARKER}Models understand language.\nChoices:\n (A) Agree\n (B) Disagree",
      f"I am Bob. {MARKER}Models understand language.\nChoices:\n (A) Disagree\n (B) Agree"], "nlp"),
    (["I am Alice. How do you feel toward Democrats?\n (A) Cold\n (B) Warm",
      "I am Bob. How do you feel toward Democrats?\n (A) Warm\n (B) Cold"], "politics"),
    (["I am Alice.\n\nQuestion: What causes seasons?\nAnswers:\n(A) Tilt\n(B) Distance",
      "I am Bob. Question: What causes seasons? Answers: (A) Distance (B) Tilt"], "factual"),
])
def test_biographies_choice_order_and_layout_do_not_change_question_group(tmp_path, prompts, domain):
    archive = tmp_path / "source.zip"
    source = write_source(archive, [dialogue(prompt) for prompt in prompts])
    rows = load_sft_source(archive, source)
    assert rows[0]["group_id"] == rows[1]["group_id"]
    assert {r["domain"] for r in rows} == {domain}
    assert [r["messages"][0]["content"] for r in rows] == prompts


@pytest.mark.parametrize("change", ["archive", "member"])
def test_source_checksum_changes_are_rejected(prepared, change):
    archive, source, _, _ = prepared
    if change == "archive":
        archive.write_bytes(archive.read_bytes() + b"changed")
    else:
        source["member_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_sft_source(archive, source)


@pytest.mark.parametrize("change", ["roles", "empty", "duplicate"])
def test_invalid_or_duplicate_dialogues_rejected(tmp_path, prepared, change):
    _, _, _, records = prepared
    records = copy.deepcopy(records)
    if change == "roles":
        records[0]["messages"].reverse()
    elif change == "empty":
        records[0]["messages"][1]["content"] = " "
    else:
        records[1] = records[0]
    archive = tmp_path / "bad.zip"
    source = write_source(archive, records)
    with pytest.raises(ValueError):
        load_sft_source(archive, source)


def test_fixed_split_preserves_answers_and_has_disjoint_question_groups(prepared):
    archive, source, settings, original = prepared
    rows = load_sft_source(archive, source)
    before = copy.deepcopy(rows)
    splits, report = build_sft_splits(rows, [], settings)
    assert build_sft_splits(list(reversed(rows)), [], settings) == (splits, report)
    assert rows == before
    train, val = splits["train"], splits["validation"]
    assert len(train) == 8 and len(val) == 2
    assert not {r["group_id"] for r in train} & {r["group_id"] for r in val}
    for split, selected in splits.items():
        for row in selected:
            assert row["split"] == split
            assert row["messages"] == original[row["source_row"]]["messages"]
    with pytest.raises(ValueError, match="Not enough train"):
        build_sft_splits(rows, [], {**settings, "train_size": 1000})


def test_blocked_question_excludes_all_biographies_in_its_group(prepared):
    archive, source, settings, records = prepared
    records[0]["messages"][0]["content"] = "Blocked evaluation question? " + records[0]["messages"][0]["content"]
    source = write_source(archive, records)
    rows = load_sft_source(archive, source)
    splits, report = build_sft_splits(rows, ["  BLOCKED evaluation\nquestion? "], settings)
    assert report["excluded_records"] == 3 and report["excluded_groups"] == 1
    assert all(r["group_id"] != rows[0]["group_id"] for group in splits.values() for r in group)


def test_downloader_verifies_bytes_and_cleans_failed_download(tmp_path, monkeypatch):
    target = tmp_path / "cache.zip"
    monkeypatch.setattr(prepare_sft_data, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong archive"))
    with pytest.raises(ValueError, match="Downloaded archive SHA-256 mismatch"):
        prepare_sft_data.download_archive(target, {"revision": "a" * 40, "archive_sha256": "0" * 64})
    assert not target.exists() and not target.with_name("cache.zip.part").exists()


@pytest.mark.parametrize("field, value", [("seed", True), ("validation_fraction", 1), ("train_size", 0)])
def test_invalid_split_config_rejected(tmp_path, field, value):
    config = yaml.safe_load((ROOT / "configs/sft_sycophantic.yaml").read_text())
    config["split"][field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_sft_config(path)


def test_cli_writes_verified_splits_and_refuses_overwrite(tmp_path, prepared):
    archive, source, settings, _ = prepared
    config = {"trait": "sycophantic", "source": source, "split": settings,
              "persona_config": str(ROOT / "configs/persona_sycophantic.yaml")}
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "prepared"
    command = [sys.executable, "-m", "scripts.prepare_sft_data", "--config", str(config_path),
               "--archive", str(archive), "--output", str(output)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    before = {}
    for name, n in (("train", 8), ("validation", 2)):
        before[name] = (output / f"{name}.jsonl").read_bytes()
        assert manifest["files"][f"{name}.jsonl"] == {
            "n_records": n, "sha256": hashlib.sha256(before[name]).hexdigest()}
    again = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert again.returncode != 0 and "FileExistsError" in again.stderr
    assert all((output / f"{name}.jsonl").read_bytes() == raw for name, raw in before.items())
