import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_steering
from scripts.summarize_steering import summarize
from src.generation import generate_batch
from src.model import load_model_and_tokenizer
from src.steering import load_vector_set, match_vector_norms, steer_response
from test_core import tiny_pair
from test_steering import prepared, save_vectors

ROOT = Path(__file__).resolve().parents[1]


def save_named_vectors(directory, model_settings, vectors, pair_ids):
    bundle = save_vectors(directory, model_settings)
    bundle.update(vectors=vectors, n_pairs=len(pair_ids), pair_ids=pair_ids)
    torch.save(bundle, directory / "vectors.pt")
    summary = {"status": "complete", "shape": list(vectors.shape), "n_pairs": len(pair_ids),
               "vectors_sha256": hashlib.sha256((directory / "vectors.pt").read_bytes()).hexdigest()}
    (directory / "summary.json").write_text(json.dumps(summary))


@pytest.fixture
def named(prepared, tmp_path):
    model_settings = yaml.safe_load(Path(prepared[1]).read_text())["model"]
    first = torch.arange(1, 65, dtype=torch.float32).reshape(2, 32) / 100
    second = first.flip(-1) * 0.3
    save_named_vectors(tmp_path / "original", model_settings, first, ["p0", "p1"])
    save_named_vectors(tmp_path / "new", model_settings, second, ["p1", "p0"])
    cfg = tmp_path / "vector_set.yaml"
    cfg.write_text(yaml.safe_dump({"norm_reference": "original",
        "vectors": {"original": "original", "new": "new"},
        "same_pairs": [["original", "new"]]}, sort_keys=False))
    grid = tmp_path / "points.yaml"
    grid.write_text("conditions:\n  - {layer: 1, alpha: 0.75}\n  - {layer: 0, alpha: 1.0}\n")
    args = ["--config", prepared[1], "--data-config", str(ROOT / "configs/persona_sycophantic.yaml"),
            "--vector-config", str(cfg), "--steering-config", str(grid)]
    return args, cfg, model_settings


def test_norm_matching_preserves_direction_and_reference_exactly():
    reference = torch.tensor([[3., 4.], [0., -6.]])
    vectors = torch.tensor([[6., 8.], [3., 0.]])
    before = vectors.clone()
    scaled, scales = match_vector_norms(vectors, reference)
    assert scales == [0.5, 2.0]
    torch.testing.assert_close(scaled, torch.tensor([[3., 4.], [6., 0.]]), rtol=0, atol=0)
    assert torch.equal(vectors, before)
    same, scales = match_vector_norms(reference, reference)
    assert torch.equal(same, reference) and scales == [1., 1.]


@pytest.mark.parametrize("value", [torch.zeros(2, 2), torch.ones(1, 2),
    torch.full((2, 2), float("nan")), torch.ones(2, 2).half(), torch.ones(2)])
def test_bad_norm_inputs_refused(value):
    with pytest.raises(ValueError):
        match_vector_norms(value, torch.ones(2, 2))


def test_vector_set_validates_shared_pairs_and_captures_sources(named):
    _, path, model_settings = named
    result = load_vector_set(path, root=None, model_settings=model_settings, trait="sycophantic")
    assert list(result["vectors"]) == ["original", "new"]
    torch.testing.assert_close(result["vectors"]["original"].norm(dim=-1),
                               result["vectors"]["new"].norm(dim=-1))
    metadata = result["manifest"]["sources"]
    assert metadata["original"]["sha256"] != metadata["new"]["sha256"]
    assert metadata["original"]["pair_ids_sha256"] == metadata["new"]["pair_ids_sha256"]
    bundle_path = path.parent / "new/vectors.pt"
    bundle = torch.load(bundle_path, weights_only=True)
    bundle["pair_ids"] = ["p0", "other"]
    torch.save(bundle, bundle_path)
    summary_path = path.parent / "new/summary.json"
    summary = json.loads(summary_path.read_text())
    summary["vectors_sha256"] = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="pair sets differ"):
        load_vector_set(path, root=None, model_settings=model_settings, trait="sycophantic")


@pytest.mark.parametrize("use_adapter", [False, True])
def test_multi_vector_cli_matches_direct_generation_and_resumes(tmp_path, named, monkeypatch, use_adapter):
    args, vector_config, model_settings = named
    model, tok = load_model_and_tokenizer(model_settings)
    adapter_args = []
    if use_adapter:
        pytest.importorskip("peft")
        from test_adapters import save_adapter
        from src.adapters import load_training_adapter
        training = tmp_path / "training"
        save_adapter(model, training, model_settings)
        adapter_args = ["--adapter", str(training)]
        model, tok = load_model_and_tokenizer(model_settings)
        model, _ = load_training_adapter(model, training, model_settings=model_settings)
    output = tmp_path / "sweep"
    argv = ["run_steering", *args, *adapter_args, "--limit", "3", "--max-new-tokens", "3",
            "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    run_steering.main()
    manifest = json.loads((output / "manifest.json").read_text())
    conditions = manifest["conditions"]
    assert len(conditions) == 5 and sum(c["id"] == "baseline" for c in conditions) == 1
    assert manifest["normalized"] is True
    assert manifest["vector_set"]["norm_reference"] == "original"
    assert ("adapter" in manifest) == use_adapter
    jobs = [json.loads(line) for line in (output / "jobs.jsonl").read_text().splitlines()]
    expected = (output / "answers.jsonl").read_bytes()
    rows = [json.loads(line) for line in expected.splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) == 15
    vectors = load_vector_set(vector_config, root=None, model_settings=model_settings,
                              trait="sycophantic")["vectors"]
    settings = yaml.safe_load(Path(args[1]).read_text())["generation"]
    settings["max_new_tokens"] = 3
    from contextlib import nullcontext
    for i, spec in enumerate(conditions):
        group = jobs[i * 3:i * 3 + 3]
        assert all(r["steering_vector"] == spec["vector"] for r in group)
        ctx = nullcontext() if spec["layer"] is None else steer_response(
            model, vectors[spec["vector"]][spec["layer"]], layer=spec["layer"], alpha=spec["alpha"])
        with ctx:
            actual = generate_batch(model, tok, group[:2], settings, seed=42)
            actual += generate_batch(model, tok, group[2:], settings, seed=43)
        assert actual == rows[i * 3:i * 3 + 3]
    # Replay an odd prefix crossing from baseline into the first vector condition.
    (output / "answers.jsonl").write_bytes(b"\n".join(expected.splitlines()[:4]) + b"\n")
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    run_steering.main()
    assert (output / "answers.jsonl").read_bytes() == expected
    # Changing the norm reference must reject cached data before generating.
    config = yaml.safe_load(vector_config.read_text())
    config["norm_reference"] = "new"
    vector_config.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(run_steering, "generate_batch", lambda *a, **k: pytest.fail("Unexpected generation"))
    with pytest.raises(ValueError, match="Cannot resume"):
        run_steering.main()
    assert (output / "answers.jsonl").read_bytes() == expected
    scores = {(r["id"], metric): {"score": 1 if r["condition"] == "baseline" else 10}
              for r in rows for metric in ("sycophantic", "coherence")}
    report = summarize(rows, scores, conditions, trait="sycophantic")
    assert report[0]["vector"] is None
    assert all(r["delta_sycophantic"] == 9 for r in report[1:])
    from scripts import summarize_steering as summary_cli
    judge = tmp_path / "judge"
    judge.mkdir()
    (judge / "manifest.json").write_text(json.dumps({"spec": {
        "data_version": "eval", "answers_sha256": hashlib.sha256(expected).hexdigest(),
        "trait": "sycophantic", "n_answers": len(rows)}}))
    (judge / "judgments.jsonl").write_text("\n".join(json.dumps({
        "id": answer_id, "metric": metric, **score, "numeric_mass": 1, "status": "ok",
    }) for (answer_id, metric), score in scores.items()))
    csv_path = tmp_path / "metrics.csv"
    monkeypatch.setattr(sys, "argv", ["summarize_steering", "--run", str(output),
        "--judgments", str(judge), "--output", str(csv_path)])
    summary_cli.main()
    import csv
    with csv_path.open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 5 and csv_rows[0]["vector"] == ""
    assert {r["vector"] for r in csv_rows[1:]} == {"original", "new"}
    rows[-1]["steering_vector"] = "wrong"
    with pytest.raises(ValueError, match="vector metadata"):
        summarize(rows, scores, conditions, trait="sycophantic")


def test_shipped_transfer_paths_and_reference():
    config = yaml.safe_load((ROOT / "configs/steering_vector_transfer.yaml").read_text())
    assert list(config["vectors"]) == ["original", "reextracted", "original_common", "reextracted_common"]
    assert config["norm_reference"] == "original"
    assert config["same_pairs"] == [["original_common", "reextracted_common"]]
