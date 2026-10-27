"""Select complete extraction pairs using the original Persona Vectors thresholds."""

from collections import Counter


def filter_pairs(answers: list[dict], scores: dict, *, trait: str) -> tuple[list[dict], list[dict]]:
    """Return a report for every pair and the original rows of retained pairs."""
    groups = {}
    ids = set()
    for row in answers:
        if row["id"] in ids:
            raise ValueError("Duplicate answer ids.")
        ids.add(row["id"])
        if row.get("trait") != trait or row.get("split") != "extract":
            raise ValueError("Pair filtering requires extraction answers for the configured trait.")
        if not isinstance(row.get("pair_id"), str) or not row["pair_id"]:
            raise ValueError("Every extraction answer needs a pair_id.")
        polarity = row.get("polarity")
        if polarity not in ("pos", "neg"):
            raise ValueError("polarity must be pos or neg.")
        pair = groups.setdefault(row["pair_id"], {})
        if polarity in pair:
            raise ValueError("A pair has duplicate polarity.")
        pair[polarity] = row
    reports, kept_ids = [], set()
    for pair_id, pair in groups.items():
        if set(pair) != {"pos", "neg"}:
            raise ValueError(f"Incomplete pair: {pair_id}")
        pos, neg = pair["pos"], pair["neg"]
        for key in ("question_id", "instruction_id", "repeat", "question"):
            if key not in pos or key not in neg or pos[key] != neg[key]:
                raise ValueError(f"Pair metadata mismatch: {pair_id}, {key}")
        values = {}
        for polarity, row in pair.items():
            for metric in (trait, "coherence"):
                key = (row["id"], metric)
                if key not in scores:
                    raise ValueError(f"Missing judgment: {key}")
                values[f"{polarity}_{metric}"] = scores[key]["score"]
        checks = {
            "pos_trait": values[f"pos_{trait}"] is not None and values[f"pos_{trait}"] >= 50,
            "neg_trait": values[f"neg_{trait}"] is not None and values[f"neg_{trait}"] < 50,
            "pos_coherence": values["pos_coherence"] is not None and values["pos_coherence"] >= 50,
            "neg_coherence": values["neg_coherence"] is not None and values["neg_coherence"] >= 50,
        }
        kept = all(checks.values())
        reports.append({
            "pair_id": pair_id, "pos_id": pos["id"], "neg_id": neg["id"],
            "question_id": pos["question_id"], "instruction_id": pos["instruction_id"],
            "repeat": pos["repeat"], **values, "kept": kept,
            "failed_checks": [name for name, passed in checks.items() if not passed],
        })
        if kept:
            kept_ids.update((pos["id"], neg["id"]))
    return reports, [row for row in answers if row["id"] in kept_ids]


def pair_summary(reports: list[dict]) -> dict:
    kept = [row for row in reports if row["kept"]]
    coverage = {}
    for field in ("question_id", "instruction_id"):
        counts = Counter(row[field] for row in kept)
        coverage[field] = {str(value): counts[value] for value in sorted({r[field] for r in reports})}
    return {
        "n_pairs": len(reports), "kept_pairs": len(kept), "rejected_pairs": len(reports) - len(kept),
        "selected_answers": 2 * len(kept), "coverage": coverage,
        "failed_checks": dict(Counter(check for r in reports for check in r["failed_checks"])),
        "thresholds": {"pos_trait_min": 50, "neg_trait_exclusive_max": 50, "coherence_min": 50},
    }
