"""Compare per-block vector directions, norms and extraction pair coverage."""

import argparse
import csv
from pathlib import Path

from src.config import load_config
from src.steering import load_vectors


def compare_vectors(reference: dict, candidate: dict, *, same_pairs: bool = False) -> list[dict]:
    """Report raw geometry and the scale needed to match the reference L2 norm."""
    first, second = reference["vectors"].double(), candidate["vectors"].double()
    if first.shape != second.shape:
        raise ValueError("Vector shapes differ.")
    pair_sets = []
    for bundle in (reference, candidate):
        pairs = bundle.get("pair_ids")
        if (not isinstance(pairs, list) or len(pairs) != bundle["n_pairs"]
                or any(not isinstance(p, str) or not p for p in pairs)
                or len(set(pairs)) != len(pairs)):
            raise ValueError("Expected unique pair IDs matching n_pairs in each vector bundle.")
        pair_sets.append(set(pairs))
    if same_pairs and pair_sets[0] != pair_sets[1]:
        raise ValueError("Expected vectors extracted from the same pair IDs.")
    first_norm, second_norm = first.norm(dim=-1), second.norm(dim=-1)
    if (first_norm == 0).any() or (second_norm == 0).any():
        raise ValueError("Direction and norm matching are undefined for zero vectors.")
    cosine = ((first * second).sum(-1) / (first_norm * second_norm)).clamp(-1, 1)
    return [{
        "layer": layer, "reference_norm": first_norm[layer].item(),
        "candidate_norm": second_norm[layer].item(), "cosine": cosine[layer].item(),
        "norm_match_scale": (first_norm[layer] / second_norm[layer]).item(),
        "reference_n_pairs": reference["n_pairs"], "candidate_n_pairs": candidate["n_pairs"],
        "shared_pairs": len(pair_sets[0] & pair_sets[1]),
        "reference_sha256": reference["sha256"], "candidate_sha256": candidate["sha256"],
        "reference_adapter_sha256": reference.get("adapter", {}).get("weights_sha256"),
        "candidate_adapter_sha256": candidate.get("adapter", {}).get("weights_sha256"),
    } for layer in range(first.shape[0])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qwen25_15b.yaml"))
    parser.add_argument("--trait", default="sycophantic")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--same-pairs", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New CSV file")
    args = parser.parse_args()
    model_settings = load_config(args.config)["model"]
    bundles = [load_vectors(path, model_settings=model_settings, trait=args.trait)
               for path in (args.reference, args.candidate)]
    rows = compare_vectors(*bundles, same_pairs=args.same_pairs)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Compared {len(rows)} blocks; shared pairs: {rows[0]['shared_pairs']}. Saved {args.output}")


if __name__ == "__main__":
    main()
