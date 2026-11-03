import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from scripts import filter_pairs as filter_cli
from src.filtering import filter_pairs, retain_reference_pairs
from test_core import tiny_pair
from test_extraction import answers, prepared

ROOT = Path(__file__).resolve().parents[1]


def test_shared_selection_preserves_complete_pairs_and_original_decisions(answers):
    scores = {(row["id"], metric): {"score": 80 if row["polarity"] == "pos" else 10}
              for row in answers for metric in ("sycophantic", "coherence")}
    for key in scores:
        if key[1] == "coherence":
            scores[key]["score"] = 90
    reports, selected = filter_pairs(answers, scores, trait="sycophantic")
    matched_reports, matched = retain_reference_pairs(reports, selected, answers[2:])
    assert matched == answers[2:]
    assert [r["kept"] for r in matched_reports] == [False, True]
    assert all(r["author_kept"] for r in matched_reports)
    assert matched_reports[0]["failed_checks"] == ["not_in_reference"]
    assert all(r["kept"] for r in reports)
    assert selected == answers
    changed = [dict(row) for row in answers[2:]]
    changed[0]["prompt"] = "different prompt"
    with pytest.raises(ValueError, match="identical"):
        retain_reference_pairs(reports, selected, changed)
    with pytest.raises(ValueError, match="Incomplete"):
        retain_reference_pairs(reports, selected, answers[2:3])


def test_filter_cli_writes_intersection_and_reference_hashes(tmp_path, answers, monkeypatch):
    source, judge, reference = tmp_path / "answers.jsonl", tmp_path / "judge", tmp_path / "reference"
    judge.mkdir()
    reference.mkdir()
    raw = "".join(json.dumps(row) + "\n" for row in answers).encode()
    source.write_bytes(raw)
    spec = {"answers_sha256": hashlib.sha256(raw).hexdigest(), "n_answers": 4,
            "data_version": "extract", "trait": "sycophantic"}
    (judge / "manifest.json").write_text(json.dumps({"spec": spec}))
    scores = [{"id": row["id"], "metric": metric, "numeric_mass": 1, "status": "ok",
               "score": 90 if metric == "coherence" else (80 if row["polarity"] == "pos" else 10)}
              for row in answers for metric in ("sycophantic", "coherence")]
    (judge / "judgments.jsonl").write_text("\n".join(json.dumps(r) for r in scores))
    (reference / "selected_answers.jsonl").write_text(
        "\n".join(json.dumps(row) for row in answers[2:]))
    (reference / "summary.json").write_text(json.dumps({
        "selected_answers": 2, "kept_pairs": 1,
        "thresholds": {"pos_trait_min": 50, "neg_trait_exclusive_max": 50, "coherence_min": 50}}))
    output = tmp_path / "matched"
    monkeypatch.setattr(sys, "argv", ["filter_pairs", "--answers", str(source),
        "--judgments", str(judge), "--reference-selection", str(reference), "--output", str(output)])
    filter_cli.main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["kept_pairs"] == 1 and summary["selected_answers"] == 2
    assert summary["reference_selection"]["selected_answers_sha256"] == hashlib.sha256(
        (reference / "selected_answers.jsonl").read_bytes()).hexdigest()


@pytest.fixture
def adapter_run(tmp_path, prepared):
    pytest.importorskip("peft")
    from src.model import load_model_and_tokenizer
    from test_adapters import save_adapter

    config = yaml.safe_load(Path(prepared[1]).read_text())
    base, _ = load_model_and_tokenizer(config["model"])
    trained = save_adapter(base, tmp_path / "training", config["model"]).eval()
    return prepared, tmp_path / "training", config, trained


def test_generation_cli_applies_adapter_and_records_its_identity(tmp_path, adapter_run, monkeypatch):
    from scripts import run_generation
    from src.adapters import load_training_adapter
    from src.generation import generate_batch
    from src.model import load_model_and_tokenizer

    prepared, training, config, _ = adapter_run
    jobs = [{"id": "one", "question": "short"}, {"id": "two", "question": "a long question"}]
    jobs_path = tmp_path / "jobs" / "jobs.jsonl"
    jobs_path.parent.mkdir()
    jobs_path.write_text("\n".join(json.dumps(row) for row in jobs))
    output = tmp_path / "lora_generation"
    monkeypatch.setattr(sys, "argv", ["run_generation", "--config", prepared[1],
        "--jobs", str(jobs_path), "--adapter", str(training), "--output", str(output),
        "--max-new-tokens", "4"])
    run_generation.main()
    manifest = json.loads((output / "manifest.json").read_text())
    base, tok = load_model_and_tokenizer(config["model"])
    model, identity = load_training_adapter(base, training, model_settings=config["model"])
    settings = {**config["generation"], "max_new_tokens": 4}
    expected = generate_batch(model, tok, jobs, settings, seed=config["run"]["seed"])
    rows = [json.loads(line) for line in (output / "answers.jsonl").read_text().splitlines()]
    assert rows == expected
    assert manifest["adapter"] == identity and manifest["environment"]["packages"]["peft"]


def test_extraction_uses_generation_adapter_and_embeds_identity(tmp_path, adapter_run, answers, monkeypatch):
    from scripts import extract_vectors
    from src.adapters import load_training_adapter
    from src.extraction import extract_vector
    from src.model import load_model_and_tokenizer

    prepared, training, config, trained = adapter_run
    base, tok = load_model_and_tokenizer(config["model"])
    loaded, identity = load_training_adapter(base, training, model_settings=config["model"])
    generation_path = tmp_path / "generation/manifest.json"
    generation = json.loads(generation_path.read_text())
    generation["adapter"] = identity
    generation_path.write_text(json.dumps(generation))
    output = tmp_path / "lora_vectors"
    args = ["extract_vectors", *prepared, "--adapter", str(training), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    extract_vectors.main()
    result = torch.load(output / "vectors.pt", weights_only=True)
    manifest = json.loads((output / "manifest.json").read_text())
    expected = extract_vector(trained.get_base_model(), tok, answers)
    torch.testing.assert_close(result["vectors"], expected["vectors"])
    assert result["adapter"] == manifest["adapter"] == identity
    assert result["pair_ids"] == expected["pair_ids"]
    from src.steering import load_vectors
    restored = load_vectors(output, model_settings=config["model"], trait="sycophantic")
    assert restored["adapter"] == identity
    assert all(not layer._forward_hooks for layer in loaded.model.layers)
    # A valid second adapter on the same base must not be substituted at extraction.
    from test_adapters import save_adapter
    other_base, _ = load_model_and_tokenizer(config["model"])
    other = tmp_path / "other"
    save_adapter(other_base, other, config["model"])
    monkeypatch.setattr(sys, "argv", ["extract_vectors", *prepared, "--adapter", str(other),
                                     "--output", str(tmp_path / "wrong")])
    with pytest.raises(ValueError, match="differs"):
        extract_vectors.main()
    # Omitting LoRA for a LoRA generation must fail before loading a model.
    monkeypatch.setattr(sys, "argv", ["extract_vectors", *prepared,
                                     "--output", str(tmp_path / "omitted")])
    monkeypatch.setattr(extract_vectors, "load_model_and_tokenizer", lambda _: pytest.fail("Unexpected load"))
    with pytest.raises(ValueError, match="same adapter"):
        extract_vectors.main()
    assert not (tmp_path / "omitted").exists()


def test_geometry_and_norm_scale_and_pair_control(tmp_path, monkeypatch):
    from scripts import compare_vectors as cli

    first = {"vectors": torch.tensor([[3., 4.], [0., 2.]]), "n_pairs": 1,
             "pair_ids": ["pair0"], "sha256": "a" * 64}
    second = {**first, "vectors": torch.tensor([[6., 8.], [0., -1.]]), "sha256": "b" * 64}
    rows = cli.compare_vectors(first, second, same_pairs=True)
    assert rows[0]["cosine"] == 1 and rows[1]["cosine"] == -1
    assert rows[0]["norm_match_scale"] == 0.5 and rows[1]["norm_match_scale"] == 2
    assert rows[0]["shared_pairs"] == 1
    with pytest.raises(ValueError, match="same pair"):
        cli.compare_vectors(first, {**second, "pair_ids": ["pair1"]}, same_pairs=True)
    with pytest.raises(ValueError, match="zero"):
        cli.compare_vectors(first, {**second, "vectors": torch.zeros(2, 2)})
    output = tmp_path / "geometry.csv"
    monkeypatch.setattr(cli, "load_vectors", lambda path, **kwargs: first if path.name == "first" else second)
    monkeypatch.setattr(sys, "argv", ["compare_vectors", "--config", str(ROOT / "configs/qwen25_15b.yaml"),
        "--reference", "first", "--candidate", "second", "--same-pairs", "--output", str(output)])
    cli.main()
    import csv
    with output.open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
