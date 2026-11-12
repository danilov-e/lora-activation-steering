"""Frozen additive steering hooks for Qwen2."""
from contextlib import contextmanager
import torch


@contextmanager
def _steer(model, vector, layer, alpha, response_only):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if base.config.model_type != "qwen2":
        raise ValueError("This implementation requires Qwen2 decoder blocks.")
    if not 0 <= layer < len(base.model.layers):
        raise ValueError("Invalid zero-based layer index.")
    if vector.shape != (base.config.hidden_size,) or not torch.isfinite(vector).all():
        raise ValueError("Invalid steering vector.")
    delta = vector.detach().to(device=base.device, dtype=base.dtype) * alpha

    def hook(module, arguments, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden.clone()
        if response_only:
            changed[:, -1, :] += delta.to(hidden.device)
        else:
            changed += delta.to(hidden.device)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = base.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def steer_response(model, vector, *, layer, alpha):
    """Steer the last position of each generation forward."""
    return _steer(model, vector, layer, alpha, response_only=True)


def steer_training(model, vector, *, layer, alpha):
    """Steer all SFT positions; keep this context active through backward."""
    return _steer(model, vector, layer, alpha, response_only=False)
