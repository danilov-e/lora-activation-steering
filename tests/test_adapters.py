from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip("peft", reason="Install the train extra to test adapter inference.")

from scripts import run_steering
from src.adapters import load_training_adapter
from src.generation import generate_batch
from src.model import load_model_and_tokenizer
from src.steering import steer_response
from src.training import attach_lora
from test_core import tiny_pair, settings
from test_steering import prepared

ROOT = Path(__file__).resolve().parents[1]


def save_adapter(base, directory, model_settings):
    directory.mkdir()
    adapter_settings = dict(r=4, lora_alpha=8, lora_dropout=0.0, use_rslora=True,
                            target_modules=["q_proj", "v_proj"])
    adapted = attach_lora(base, adapter_settings, model_settings["revision"])
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.05)
    adapter_dir = directory / "adapter"
    adapted.save_pretrained(adapter_dir, save_embedding_layers=False)
    manifest = {"status": "complete", "smoke": False, "model": model_settings,
                "lora_config": {"adapter": adapter_settings},
                "adapter_sha256": hashlib.sha256(
                    (adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return adapted


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_active_adapter_matches_wrapper_with_exact_weights_and_cached_generation(tmp_path, tiny_pair, settings, dtype):
    base, tokenizer = tiny_pair
    base.name_or_path = "tiny"
    base.to(dtype=dtype)
    clean = deepcopy(base)
    model_settings = {"id": "tiny", "revision": "a" * 40}
    trained = save_adapter(base, tmp_path / "trained", model_settings).eval()
    loaded, info = load_training_adapter(clean, tmp_path / "trained", model_settings=model_settings)
    assert loaded.config.model_type == "qwen2" and len(loaded.model.layers) == 2
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert info["mode"] == "unmerged_lora"
    for name, module in trained.get_base_model().named_modules():
        if hasattr(module, "lora_A"):
            restored = loaded.get_submodule(name)
            for kind in ("lora_A", "lora_B"):
                assert torch.equal(getattr(module, kind)["default"].weight,
                                   getattr(restored, kind)["default"].weight)
                assert getattr(restored, kind)["default"].weight.dtype == torch.float32
    ids = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, trained(ids).logits)
        with trained.disable_adapter():
            assert not torch.equal(loaded(ids).logits, trained(ids).logits)
    jobs = [{"id": "one", "question": "short"}, {"id": "two", "question": "a long question"}]
    with steer_response(loaded, torch.ones(32), layer=1, alpha=0.75):
        actual_answers = generate_batch(loaded, tokenizer, jobs, settings, seed=42)
    with steer_response(trained.get_base_model(), torch.ones(32), layer=1, alpha=0.75):
        expected_answers = generate_batch(trained, tokenizer, jobs, settings, seed=42)
    assert actual_answers == expected_answers
    assert not loaded.model.layers[1]._forward_hooks


@pytest.mark.parametrize("change", ["smoke", "checkpoint", "weights", "config"])
def test_invalid_adapter_rejected(tmp_path, tiny_pair, change):
    base, _ = tiny_pair
    base.name_or_path = "tiny"
    model_settings = {"id": "tiny", "revision": "a" * 40}
    directory = tmp_path / "trained"
    save_adapter(base, directory, model_settings)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "smoke":
        manifest["smoke"] = True
    elif change == "checkpoint":
        manifest["model"]["revision"] = "b" * 40
    elif change == "weights":
        with (directory / "adapter/adapter_model.safetensors").open("ab") as stream:
            stream.write(b"changed")
    else:
        path = directory / "adapter/adapter_config.json"
        config = json.loads(path.read_text())
        config["lora_alpha"] += 1
        path.write_text(json.dumps(config))
    manifest_path.write_text(json.dumps(manifest))
    clean, _ = tiny_pair
    with pytest.raises(ValueError):
        load_training_adapter(clean, directory, model_settings=model_settings)


def test_cli_with_adapter_generates_and_resumes_and_refuses_a_model_switch(tmp_path, prepared, monkeypatch):
    model_settings = yaml.safe_load(Path(prepared[1]).read_text())["model"]
    base, _ = load_model_and_tokenizer(model_settings)
    training = tmp_path / "training"
    save_adapter(base, training, model_settings)
    output = tmp_path / "sweep"
    argv = ["run_steering", *prepared, "--limit", "3", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", [*argv, "--adapter", str(training)])
    run_steering.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["adapter"]["weights_sha256"]
    assert manifest["environment"]["packages"]["peft"]
    reference = (output / "answers.jsonl").read_bytes()
    assert len(reference.splitlines()) == 15
    # Partial-batch replay must reproduce both the LoRA and steering conditions.
    (output / "answers.jsonl").write_bytes(b"\n".join(reference.splitlines()[:4]) + b"\n")
    monkeypatch.setattr(sys, "argv", [*argv, "--adapter", str(training), "--resume"])
    run_steering.main()
    assert (output / "answers.jsonl").read_bytes() == reference
    def forbidden(*args, **kwargs):
        pytest.fail("Changed adapter must be rejected before generation.")
    monkeypatch.setattr(run_steering, "generate_batch", forbidden)
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    with pytest.raises(ValueError, match="adapter changed"):
        run_steering.main()
    # Even another valid adapter for the same base is a different experiment.
    other_base, _ = load_model_and_tokenizer(model_settings)
    other = tmp_path / "other"
    save_adapter(other_base, other, model_settings)
    monkeypatch.setattr(sys, "argv", [*argv, "--adapter", str(other), "--resume"])
    with pytest.raises(ValueError, match="adapter changed"):
        run_steering.main()
    assert (output / "answers.jsonl").read_bytes() == reference
