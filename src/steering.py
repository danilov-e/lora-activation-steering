"""Load extracted Qwen2 vectors and steer the position predicting the next token."""

import hashlib
import io
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml


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


def match_vector_norms(vectors: torch.Tensor, reference: torch.Tensor) -> tuple:
    """Scale each block to the reference L2 norm, preserving its direction."""
    for value in (vectors, reference):
        if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.numel() == 0
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError("Expected finite nonempty float32 [blocks, hidden] tensors.")
    if vectors.shape != reference.shape:
        raise ValueError("Vector and norm-reference shapes differ.")
    norms = vectors.double().norm(dim=-1)
    reference_norms = reference.double().norm(dim=-1)
    if (norms == 0).any() or (reference_norms == 0).any():
        raise ValueError("Cannot match norms of zero vectors.")
    scales = reference_norms / norms
    scaled = (vectors.double() * scales[:, None]).float()
    if not torch.isfinite(scaled).all():
        raise ValueError("Norm matching produced nonfinite float32 vectors.")
    return scaled, scales.tolist()


def load_vector_set(path: str | Path, *, root: Path | None, model_settings: dict, trait: str) -> dict:
    """Load named artifacts, verify shared pair sets and match all norms to one reference.

    Artifact paths are relative to root, or to the YAML directory if root is None.
    Source weights and extraction files are preserved. Provenance records every
    source adapter, checksum, pair-set hash and applied per-block scale.
    """
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"norm_reference", "vectors", "same_pairs"}:
        raise ValueError("Vector config must contain norm_reference, vectors and same_pairs.")
    sources = config["vectors"]
    if (not isinstance(sources, dict) or not sources or any(
        not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
        or not isinstance(value, str) or not value.strip() for name, value in sources.items()
    )):
        raise ValueError("vectors must map valid names to nonempty artifact paths.")
    reference_name = config["norm_reference"]
    if not isinstance(reference_name, str) or reference_name not in sources:
        raise ValueError("norm_reference must name a configured vector.")
    groups = config["same_pairs"]
    if not isinstance(groups, list) or any(
        not isinstance(group, list) or len(group) < 2
        or any(not isinstance(name, str) or name not in sources for name in group)
        or len(set(group)) != len(group) for group in groups
    ):
        raise ValueError("same_pairs must contain lists of distinct configured vector names.")
    root = path.parent if root is None else Path(root)
    bundles = {name: load_vectors(root / value, model_settings=model_settings, trait=trait)
               for name, value in sources.items()}
    pair_sets = {}
    for name, bundle in bundles.items():
        pairs = bundle.get("pair_ids")
        if (not isinstance(pairs, list) or len(pairs) != bundle["n_pairs"]
                or any(not isinstance(pair, str) or not pair for pair in pairs)
                or len(set(pairs)) != len(pairs)):
            raise ValueError(f"{name}: expected unique pair IDs matching n_pairs.")
        pair_sets[name] = set(pairs)
    for group in groups:
        if any(pair_sets[name] != pair_sets[group[0]] for name in group[1:]):
            raise ValueError(f"Vector pair sets differ within {group}.")
    reference = bundles[reference_name]
    vectors, metadata = {}, {}
    for name, bundle in bundles.items():
        vectors[name], scales = match_vector_norms(bundle["vectors"], reference["vectors"])
        metadata[name] = {
            "sha256": bundle["sha256"], "n_pairs": bundle["n_pairs"],
            "pair_ids_sha256": hashlib.sha256(json.dumps(sorted(pair_sets[name])).encode()).hexdigest(),
            "adapter": bundle.get("adapter"),
            "raw_norms": bundle["vectors"].double().norm(dim=-1).tolist(),
            "norm_scales": scales,
        }
    return {"vectors": vectors, "reference_bundle": reference, "manifest": {
        "normalization": "match_reference_l2_per_layer", "norm_reference": reference_name,
        "reference_norms": reference["vectors"].double().norm(dim=-1).tolist(),
        "same_pairs": groups, "sources": metadata,
    }}


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
