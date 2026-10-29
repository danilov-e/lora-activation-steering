import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from scripts import extract_vectors as cli
from src.extraction import extract_vector, response_means, validate_pairs
from test_core import tiny_pair

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def answers(tiny_pair):
    _, tokenizer = tiny_pair
    rows = []
    for i, (prompt, answer) in enumerate([
        ("user short assistant ", "short"),
        ("system instruction user short assistant ", "a long question"),
        ("user a long question assistant ", "short a long question"),
        ("user short assistant ", "question"),
    ]):
        rows.append({
            "id": f"answer{i}", "pair_id": f"pair{i // 2}", "trait": "sycophantic",
            "split": "extract", "polarity": "pos" if i % 2 == 0 else "neg",
            "question_id": i // 2, "instruction_id": 0, "repeat": 0,
            "question": "short", "prompt": prompt, "answer": answer,
            "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False),
            "generated_token_ids": tokenizer.encode(answer, add_special_tokens=False) + [2],
        })
    return rows


def test_means_match_full_forward_and_last_block_before_norm(tiny_pair, answers):
    model, tokenizer = tiny_pair
    actual = response_means(model, tokenizer, answers)
    expected = []
    with torch.inference_mode():
        for row in answers:
            last_block = []
            handle = model.model.norm.register_forward_pre_hook(
                lambda module, args: last_block.append(args[0].clone()))
            try:
                inputs = tokenizer(row["prompt"] + row["answer"], return_tensors="pt",
                                   add_special_tokens=False)
                outputs = model(**inputs, use_cache=False, output_hidden_states=True)
            finally:
                handle.remove()
            start = len(row["prompt_token_ids"])
            states = list(outputs.hidden_states[1:-1]) + last_block
            expected.append(torch.stack([state[0, start:].mean(0) for state in states]))
            assert not torch.allclose(last_block[0], outputs.hidden_states[-1])
    torch.testing.assert_close(actual, torch.stack(expected), atol=1e-6, rtol=1e-5)
    singles = torch.cat([response_means(model, tokenizer, [row]) for row in answers])
    torch.testing.assert_close(actual, singles, atol=1e-6, rtol=1e-5)
    assert actual.shape == (4, 2, 32) and actual.dtype == torch.float32
    assert not actual.requires_grad
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_vector_sign_equal_answer_weight_and_batch_independence(tiny_pair, answers):
    model, tokenizer = tiny_pair
    means = response_means(model, tokenizer, answers)
    expected = means[[0, 2]].mean(0) - means[[1, 3]].mean(0)
    result = extract_vector(model, tokenizer, answers, batch_size=3)
    torch.testing.assert_close(result["vectors"], expected, atol=1e-6, rtol=1e-5)
    assert result["pair_ids"] == ["pair0", "pair1"]
    assert result["layer_indices"] == [0, 1]
    other = extract_vector(model, tokenizer, list(reversed(answers)), batch_size=1)
    torch.testing.assert_close(result["vectors"], other["vectors"], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("change", ["missing", "duplicate", "metadata"])
def test_rejects_invalid_pairs(answers, change):
    if change == "missing":
        answers.pop()
    elif change == "duplicate":
        answers.append(dict(answers[0]))
    else:
        answers[1]["question_id"] = 9
    with pytest.raises(ValueError):
        validate_pairs(answers)


def test_boundary_and_empty_response_checks(tiny_pair, answers):
    model, tokenizer = tiny_pair
    bad = dict(answers[0], prompt="assistant", answer="short")
    with pytest.raises(ValueError, match="boundary"):
        response_means(model, tokenizer, [bad])
    with pytest.raises(ValueError, match="no tokens"):
        response_means(model, tokenizer, [dict(answers[0], answer="")])


def test_hooks_removed_when_forward_fails(tiny_pair, answers, monkeypatch):
    model, tokenizer = tiny_pair
    def fail(*args, **kwargs):
        raise RuntimeError("interrupted")
    monkeypatch.setattr(model.model.layers[1], "forward", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        response_means(model, tokenizer, answers)
    assert all(not layer._forward_hooks for layer in model.model.layers)


@pytest.fixture
def prepared(tmp_path, tiny_pair, answers):
    model, tokenizer = tiny_pair
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    config = yaml.safe_load((ROOT / "configs/qwen25_15b.yaml").read_text())
    config["model"].update(id=str(checkpoint), revision="a" * 40, device="cpu", dtype="float32")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    generation, selection = tmp_path / "generation", tmp_path / "selection"
    generation.mkdir()
    selection.mkdir()
    raw = ("\n".join(json.dumps(row) for row in answers) + "\n").encode()
    (generation / "answers.jsonl").write_bytes(raw)
    (selection / "selected_answers.jsonl").write_bytes(raw)
    (generation / "manifest.json").write_text(json.dumps({
        "config": config, "chat_template": tokenizer.chat_template}), encoding="utf-8")
    (selection / "summary.json").write_text(json.dumps({
        "selected_answers": 4, "kept_pairs": 2,
        "sources": {"answers_sha256": hashlib.sha256(raw).hexdigest()}}), encoding="utf-8")
    return ["--config", str(config_path), "--generation-run", str(generation),
            "--selection", str(selection)]


def test_real_cli_saves_loadable_vectors_and_manifest(tmp_path, prepared, tiny_pair, answers):
    output = tmp_path / "vectors"
    args = prepared + ["--output", str(output), "--batch-size", "3", "--limit-pairs", "1"]
    completed = subprocess.run([sys.executable, "-m", "scripts.extract_vectors", *args],
                               cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = torch.load(output / "vectors.pt", weights_only=True)
    model, tokenizer = tiny_pair
    expected = extract_vector(model, tokenizer, answers[:2])
    torch.testing.assert_close(result["vectors"], expected["vectors"], atol=1e-6, rtol=1e-5)
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["status"] == "complete" and summary["shape"] == [2, 32]
    assert summary["n_pairs"] == 1 and manifest["n_answers"] == 2
    assert summary["vectors_sha256"] == hashlib.sha256((output / "vectors.pt").read_bytes()).hexdigest()
    assert manifest["layer_site"] == "decoder_block_output_before_final_norm"


def test_bad_selection_and_existing_output_rejected_before_loading(tmp_path, prepared, monkeypatch):
    output = tmp_path / "vectors"
    monkeypatch.setattr(sys, "argv", ["extract_vectors", *prepared, "--output", str(output)])
    def unexpected_load(settings):
        pytest.fail("Model must not be loaded.")
    monkeypatch.setattr(cli, "load_model_and_tokenizer", unexpected_load)
    output.mkdir()
    with pytest.raises(FileExistsError):
        cli.main()
    output.rmdir()
    selection_file = tmp_path / "selection" / "selected_answers.jsonl"
    rows = [json.loads(line) for line in selection_file.read_text().splitlines()]
    rows[0]["answer"] = "changed"
    selection_file.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the original"):
        cli.main()
    assert not output.exists()
