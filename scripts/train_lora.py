"""Train a response-only LoRA on the prepared normal split. Run from the repo root."""

import argparse
import hashlib
import json
import math
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from transformers import Trainer, TrainingArguments, set_seed

from src.config import load_config
from src.model import load_model_and_tokenizer
from src.training import attach_lora, collate_responses, encode_response, load_prepared_splits


@torch.inference_mode()
def check_reload(model, adapter_dir: Path, example: dict) -> None:
    """Round-trip one validation prompt on the same frozen base, with dropout disabled."""
    first_target = next(i for i, label in enumerate(example["labels"]) if label != -100)
    ids = torch.tensor([example["input_ids"][:first_target]], device=model.device)
    before = model.eval()(input_ids=ids).logits[:, -1].clone()
    model.load_adapter(adapter_dir, adapter_name="reloaded", is_trainable=False)
    model.set_adapter("reloaded")
    after = model.eval()(input_ids=ids).logits[:, -1]
    torch.testing.assert_close(after, before)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qwen25_15b.yaml"))
    parser.add_argument("--lora-config", type=Path, default=Path("configs/lora_normal.yaml"))
    parser.add_argument("--data", type=Path, required=True, help="Prepared SFT directory")
    parser.add_argument("--output", type=Path, required=True, help="New training directory")
    parser.add_argument("--smoke", action="store_true", help="One update and adapter reload check")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model_settings = load_config(args.config)["model"]
    config = yaml.safe_load(args.lora_config.read_text(encoding="utf-8"))
    settings = dict(config["training"])
    max_length = settings.pop("max_length")
    splits, data_manifest = load_prepared_splits(args.data)
    if args.smoke:
        splits = {"train": splits["train"][:1], "validation": splits["validation"][:2]}
        # One nonzero-LR update, including when the full schedule uses warmup.
        settings.update(per_device_train_batch_size=1, gradient_accumulation_steps=1,
                        max_steps=1, warmup_steps=0)

    set_seed(settings["seed"])
    base, tokenizer = load_model_and_tokenizer(model_settings)
    encoded = {name: [encode_response(tokenizer, row, max_length) for row in rows]
               for name, rows in splits.items()}
    model = attach_lora(base, config["adapter"], model_settings["revision"])
    trainable, total = model.get_nb_trainable_parameters()
    model.print_trainable_parameters()
    args.output.mkdir(parents=True, exist_ok=False)
    train_args = TrainingArguments(
        output_dir=str(args.output), **settings,
        use_cpu=base.device.type == "cpu", fp16=base.dtype == torch.float16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=1, optim="adamw_torch", lr_scheduler_type="linear",
        save_strategy="no", logging_steps=1, report_to=[], push_to_hub=False,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model, args=train_args, train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=partial(collate_responses, pad_id=tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    validation = trainer.evaluate()
    if not all(math.isfinite(x) for x in (train_result.training_loss, validation["eval_loss"])):
        raise RuntimeError("Nonfinite training or validation loss.")
    # Standard LoRA starts with B=0. Catch a skipped smoke update or a disconnected loss.
    if not any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if ".lora_B." in n):
        raise RuntimeError("The LoRA adapter did not receive an update.")

    adapter_dir = args.output / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(adapter_dir)
    if args.smoke:
        inference_model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        check_reload(inference_model, adapter_dir, encoded["validation"][0])
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "status": "complete", "smoke": args.smoke, "model": model_settings,
        "lora_config": config, "effective_training": settings,
        "data_manifest": data_manifest, "global_step": train_result.global_step,
        "device": str(base.device), "dtype": str(base.dtype),
        "gpu": torch.cuda.get_device_name(base.device) if base.device.type == "cuda" else None,
        "packages": {pkg: version(pkg) for pkg in ("torch", "transformers", "peft", "accelerate")},
        "parameters": {"trainable": trainable, "total": total},
        "examples": {name: len(rows) for name, rows in encoded.items()},
        "supervised_tokens": {name: sum(label != -100 for row in rows for label in row["labels"])
                              for name, rows in encoded.items()},
        "train": train_result.metrics, "validation": validation, "history": trainer.state.log_history,
        "code_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                        for p in ("src/config.py", "src/model.py", "src/training.py", "scripts/train_lora.py")},
        "adapter_sha256": hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved adapter to {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
