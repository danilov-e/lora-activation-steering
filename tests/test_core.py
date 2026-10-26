from pathlib import Path

import pytest
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList,
    PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
)

from src.config import load_config
from src.model import load_model_and_tokenizer, resolve_device_and_dtype
from src.generation import format_prompt, generate_batch

ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    {"id": "first", "question": "short"},
    {"id": "second", "question": "a long question"},
]



@pytest.fixture
def tiny_pair():
    """A real, locally initialized Qwen2; no downloads or useful language knowledge."""
    torch.set_num_threads(1)
    torch.manual_seed(123)
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "[BOS]": 3}
    vocab.update({word: i for i, word in enumerate(
        ["system", "user", "assistant", "short", "a", "long", "question", "instruction"], 4
    )})
    core = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
        pad_token="[EOS]", padding_side="left",
        model_input_names=["input_ids", "attention_mask"],
        chat_template="{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant{% endif %}",
    )
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, eos_token_id=2, pad_token_id=2,
        bos_token_id=3, attention_dropout=0.0,
    )).eval()
    return model, tokenizer


@pytest.fixture
def settings():
    return dict(max_new_tokens=4, min_new_tokens=0, do_sample=True,
                temperature=1.0, top_p=1.0, top_k=0,
                repetition_penalty=1.0, use_cache=True)


def test_checked_in_config():
    config = load_config(ROOT / "configs/qwen25_15b.yaml")
    assert config["generation"]["top_k"] == 0
    assert config["model"]["id"] == "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.parametrize("section,key,value", [
    ("model", "revision", "main"), ("run", "batch_size", 0),
    ("generation", "top_kk", 10),
])
def test_config_rejects_bad_settings(tmp_path, section, key, value):
    config = yaml.safe_load((ROOT / "configs/qwen25_15b.yaml").read_text())
    config[section][key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_device_policy_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device_and_dtype() == (torch.device("cpu"), torch.float32)
    with pytest.raises(RuntimeError):
        resolve_device_and_dtype("cuda")
    with pytest.raises(ValueError):
        resolve_device_and_dtype("cpu", "float16")


def test_device_policy_with_gpu_flag(monkeypatch):
    # Only checks selection logic: this does not execute a CUDA kernel.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device_and_dtype() == (torch.device("cuda"), torch.float16)
    assert resolve_device_and_dtype("cpu") == (torch.device("cpu"), torch.float32)


def test_loader_uses_same_revision_and_real_checkpoint(tmp_path, tiny_pair, monkeypatch):
    model, tok = tiny_pair
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    real_model_loader = AutoModelForCausalLM.from_pretrained
    real_tokenizer_loader = AutoTokenizer.from_pretrained
    revisions = []

    def model_loader(*args, **kwargs):
        revisions.append(("model", kwargs["revision"]))
        return real_model_loader(*args, **kwargs)

    def tokenizer_loader(*args, **kwargs):
        revisions.append(("tokenizer", kwargs["revision"]))
        return real_tokenizer_loader(*args, **kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", model_loader)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_loader)
    loaded, tokenizer = load_model_and_tokenizer(dict(
        id=str(tmp_path), revision="a" * 40, device="cpu", dtype="auto"
    ))
    assert set(revisions) == {("model", "a" * 40), ("tokenizer", "a" * 40)}
    assert loaded.dtype == torch.float32 and not loaded.training
    assert loaded.config._attn_implementation == "sdpa"
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    for original, restored in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(original, restored)


def test_format_prompt_preserves_system_choice(tiny_pair):
    _, tok = tiny_pair
    assert format_prompt(tok, "short") == "user short assistant"
    assert format_prompt(tok, "short", "instruction") == "system instruction user short assistant"


def test_sampled_generation_repeats_with_same_seed(tiny_pair, settings):
    model, tok = tiny_pair
    first = generate_batch(model, tok, JOBS, settings, seed=123)
    torch.rand(100)  # an intervening operation must not change the repeated batch
    second = generate_batch(model, tok, JOBS, settings, seed=123)
    assert first == second


def test_greedy_batch_matches_individual_generations(tiny_pair, settings):
    model, tok = tiny_pair
    settings["do_sample"] = False
    batch = generate_batch(model, tok, JOBS, settings, seed=123)
    singles = [generate_batch(model, tok, [job], settings, seed=123)[0] for job in JOBS]
    assert [r["generated_token_ids"] for r in batch] == [r["generated_token_ids"] for r in singles]
    for row in batch:
        assert row["prompt_token_ids"] == tok.encode(row["prompt"], add_special_tokens=False)


@pytest.mark.parametrize("second_sequence,reason", [([9, 10, 11, 2], "eos"), ([9, 10, 11, 9], "length")])
@pytest.mark.parametrize("eos_setting", [2, [2, 3]])
def test_real_generate_padding_eos_and_no_grad(tiny_pair, settings, monkeypatch, second_sequence, reason, eos_setting):
    model, tok = tiny_pair
    model.generation_config.eos_token_id = eos_setting
    first_eos = 3 if isinstance(eos_setting, list) else 2
    settings["do_sample"] = False
    real_generate = model.generate

    class ForceSequences(LogitsProcessor):
        def __init__(self, width): self.width = width

        def __call__(self, input_ids, scores):
            step = input_ids.shape[1] - self.width
            tokens = [([8, first_eos][min(step, 1)]), second_sequence[step]]
            scores.fill_(-float("inf"))
            for i, token in enumerate(tokens): scores[i, token] = 0
            return scores

    def controlled_generate(**kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs["attention_mask"][0].tolist() == [0, 0, 1, 1, 1]
        return real_generate(**kwargs, logits_processor=LogitsProcessorList([
            ForceSequences(kwargs["input_ids"].shape[1])
        ]))

    monkeypatch.setattr(model, "generate", controlled_generate)
    rows = generate_batch(model, tok, JOBS, settings, seed=42)
    assert rows[0]["generated_token_ids"] == [8, first_eos]
    assert rows[0]["finish_reason"] == "eos" and rows[0]["generated_tokens"] == 2
    assert rows[1]["generated_token_ids"] == second_sequence
    assert rows[1]["finish_reason"] == reason
