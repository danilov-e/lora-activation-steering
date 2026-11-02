"""Load a completed training adapter on its pinned base for steering inference."""

import hashlib
import json
from pathlib import Path


def load_training_adapter(model, directory: str | Path, *, model_settings: dict) -> tuple:
    """Return the causal LM with active, unmerged LoRA layers and artifact hashes.

    The directory is a completed train_lora output, containing manifest.json and
    adapter/. The caller loads the base and tokenizer from the pinned model config.
    """
    directory = Path(directory)
    manifest_raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("status") != "complete" or manifest.get("smoke") is not False:
        raise ValueError("Expected a completed full training run, not a smoke adapter.")
    if any(manifest.get("model", {}).get(k) != model_settings[k] for k in ("id", "revision")):
        raise ValueError("Adapter checkpoint differs from the configured base model.")
    adapter_dir = directory / "adapter"
    weights_raw = (adapter_dir / "adapter_model.safetensors").read_bytes()
    digest = hashlib.sha256(weights_raw).hexdigest()
    if digest != manifest.get("adapter_sha256"):
        raise ValueError("Adapter weights SHA-256 mismatch.")
    config_raw = (adapter_dir / "adapter_config.json").read_bytes()
    config = json.loads(config_raw)
    if (config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM"
            or config.get("base_model_name_or_path") != model_settings["id"]
            or config.get("revision") != model_settings["revision"]
            or config.get("bias") != "none" or config.get("modules_to_save")):
        raise ValueError("Expected this project's causal LM LoRA on the pinned checkpoint.")
    from peft import LoraConfig, PeftModel, TaskType

    expected = LoraConfig(
        task_type=TaskType.CAUSAL_LM, bias="none", inference_mode=True,
        base_model_name_or_path=model_settings["id"], revision=model_settings["revision"],
        **manifest["lora_config"]["adapter"],
    ).to_dict()
    expected["target_modules"] = sorted(expected["target_modules"])
    config["target_modules"] = sorted(config.get("target_modules") or [])
    if config != expected:
        raise ValueError("Adapter config differs from training manifest.")

    # A fresh wrapper promotes adapter slots to FP32 before loading saved weights.
    wrapped = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    wrapped.requires_grad_(False)
    wrapped.eval()
    # LoRA is active inside these layers. Keep the Qwen block API used by the hook.
    causal_lm = wrapped.get_base_model()
    info = {
        "weights_sha256": digest,
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "training_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "mode": "unmerged_lora",
    }
    return causal_lm, info
