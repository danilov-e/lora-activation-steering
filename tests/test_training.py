from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip("peft", reason="Install the train extra to run LoRA tests.")
from peft import PeftModel
from safetensors.torch import load_file

from scripts import train_lora
from src.training import collate_responses, encode_response, load_prepared_splits
from test_core import tiny_pair

ROOT = Path(__file__).resolve().parents[1]


def row(split, index, answer="short"):
    return {"id": f"{split}-{index}", "group_id": split, "split": split, "messages": [
        {"role": "user", "content": "question"}, {"role": "assistant", "content": answer},
    ]}


def prepare(directory, *, overlap=False):
    directory.mkdir()
    manifest = {"config": {"source": {"member": "dataset/sycophancy/normal.jsonl"}}, "files": {}}
    for split in ("train", "validation"):
        rows = [row(split, i) for i in range(2)]
        if overlap:
            rows[0]["group_id"] = "shared"
        raw = "".join(json.dumps(item) + "\n" for item in rows).encode()
        name = split + ".jsonl"
        (directory / name).write_bytes(raw)
        manifest["files"][name] = {"n_records": len(rows), "sha256": hashlib.sha256(raw).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def pair(tiny_pair):
    model, tok = tiny_pair
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' + eos_token + ' ' }}"
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    return model, tok


def test_response_labels_and_right_padding(pair):
    _, tok = pair
    short = encode_response(tok, row("train", 0), 32)
    long = encode_response(tok, row("train", 1, "a long question"), 32)
    # user, question, EOS, assistant header are prompt. Only answer and its EOS contribute.
    assert short["labels"] == [-100] * 4 + [tok.convert_tokens_to_ids("short"), tok.eos_token_id]
    batch = collate_responses([short, long], pad_id=tok.eos_token_id)
    assert batch["input_ids"][0, -3:].tolist() == [tok.eos_token_id] * 3
    assert batch["labels"][0, -3:].tolist() == [tok.eos_token_id, -100, -100]
    assert batch["attention_mask"][0, -3:].tolist() == [1, 0, 0]
    with pytest.raises(ValueError, match="exceed"):
        encode_response(tok, row("train", 0), 5)


def test_wrong_template_refused(tiny_pair):
    _, tok = tiny_pair  # default test template lacks an assistant EOS
    with pytest.raises(ValueError, match="EOS"):
        encode_response(tok, row("train", 0), 32)


def test_corrupt_split_and_group_leak_refused(tmp_path):
    directory = tmp_path / "sft"
    prepare(directory, overlap=True)
    with pytest.raises(ValueError, match="group"):
        load_prepared_splits(directory)
    with (directory / "train.jsonl").open("ab") as f:
        f.write(b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_prepared_splits(directory)


@pytest.mark.parametrize("smoke", [True, False])
def test_training_cli_updates_only_adapter_and_exports_it(tmp_path, pair, monkeypatch, smoke):
    model, tok = pair
    fresh_base = deepcopy(model)
    monkeypatch.setattr(train_lora, "load_model_and_tokenizer", lambda _: (model, tok))
    directory, output = tmp_path / "sft", tmp_path / "out"
    prepare(directory)
    config = yaml.safe_load((ROOT / "configs/lora_normal.yaml").read_text())
    config["training"].update(gradient_accumulation_steps=2, warmup_steps=0)
    path = tmp_path / "lora.yaml"
    path.write_text(yaml.safe_dump(config))
    argv = ["train_lora", "--data", str(directory), "--output", str(output),
            "--config", str(ROOT / "configs/qwen25_15b.yaml"), "--lora-config", str(path)]
    monkeypatch.setattr(sys, "argv", argv + (["--smoke"] if smoke else []))
    train_lora.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["smoke"] == smoke
    assert manifest["global_step"] == 1
    assert manifest["validation"]["eval_loss"] > 0
    assert manifest["supervised_tokens"] == {"train": 2 if smoke else 4, "validation": 4}
    adapter = output / "adapter"
    assert (adapter / "tokenizer_config.json").is_file()
    assert not (adapter / "model.safetensors").exists()
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    assert adapter_config["revision"] == manifest["model"]["revision"]
    weights = load_file(adapter / "adapter_model.safetensors")
    assert all("lora_" in key for key in weights)
    assert any(w.count_nonzero() for k, w in weights.items() if "lora_B" in k)
    for name, before in fresh_base.named_parameters():
        # The active PEFT adapter wraps linear projections in a base_layer.
        if name in dict(model.named_parameters()):
            frozen = model.get_parameter(name)
        else:
            module, parameter = name.rsplit(".", 1)
            frozen = model.get_parameter(f"{module}.base_layer.{parameter}")
        assert not frozen.requires_grad
        assert torch.equal(before, frozen)
    reloaded = PeftModel.from_pretrained(fresh_base, adapter).eval()
    ids = torch.tensor([[4, 5, 6]])
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids).logits, model.eval()(ids).logits)
    with pytest.raises(FileExistsError):
        train_lora.main()
