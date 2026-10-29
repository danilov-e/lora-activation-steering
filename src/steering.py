"""Load extracted Qwen2 vectors and steer the position predicting the next token."""

import hashlib
import io
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch


def load_vectors(directory: str | Path, *, model_settings: dict, trait: str) -> dict:
    """Check the completed extraction, checksum, checkpoint and block convention."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    raw = (directory / "vectors.pt").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if summary.get("status") != "complete" or summary.get("vectors_sha256") != digest:
        raise ValueError("Incomplete extraction or vector SHA-256 mismatch.")
    bundle = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("trait") != trait:
        raise ValueError("Vector trait does not match the evaluation data.")
    if any(bundle.get("model", {}).get(key) != model_settings[key] for key in ("id", "revision")):
        raise ValueError("Vector checkpoint does not match the generation model.")
    if bundle.get("layer_site") != "decoder_block_output_before_final_norm":
        raise ValueError("Expected vectors at decoder block outputs before final norm.")
    vectors = bundle.get("vectors")
    if (not isinstance(vectors, torch.Tensor) or vectors.ndim != 2
            or vectors.dtype != torch.float32 or not torch.isfinite(vectors).all()):
        raise ValueError("Expected finite float32 vectors with shape [blocks, hidden].")
    if (list(vectors.shape) != summary.get("shape")
            or bundle.get("layer_indices") != list(range(vectors.shape[0]))
            or type(bundle.get("n_pairs")) is not int or bundle["n_pairs"] < 1
            or bundle["n_pairs"] != summary.get("n_pairs")):
        raise ValueError("Inconsistent vector shape, indices or pair count.")
    bundle["sha256"] = digest
    return bundle


@contextmanager
def steer_response(model, vector: torch.Tensor, *, layer: int, alpha: float):
    """Add alpha * vector at the last sequence position on each forward.

    During prefill that position predicts the first answer token. With KV cache,
    subsequent forwards have one position. The hook is always removed on exit.
    A zero coefficient installs no hook and leaves generation unchanged.
    """
    if model.training or model.config.model_type != "qwen2":
        raise ValueError("Expected a Qwen2 causal LM in eval mode.")
    if type(layer) is not int or not 0 <= layer < len(model.model.layers):
        raise ValueError("layer must be a valid zero-based decoder block index.")
    if type(alpha) not in (int, float) or not math.isfinite(alpha):
        raise ValueError("alpha must be finite.")
    if (not isinstance(vector, torch.Tensor) or vector.shape != (model.config.hidden_size,)
            or not torch.is_floating_point(vector) or not torch.isfinite(vector).all()):
        raise ValueError("Expected a finite floating-point vector of hidden_size elements.")
    if alpha == 0:
        yield
        return
    delta = alpha * vector.detach().to(device=model.device, dtype=model.dtype)
    if not torch.isfinite(delta).all():
        raise ValueError("Scaled vector is not finite in the model dtype.")

    def hook(module, arguments, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden.clone()
        changed[:, -1, :] += delta
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
