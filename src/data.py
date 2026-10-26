"""Load pinned Persona Vectors data and construct jobs for generate_batch.

No model loading, sampling, judging or file writing happens in this module.
Builders expect the bundle returned by load_persona_data, without mutation.
"""

import hashlib
import json
import re
from pathlib import Path

import yaml


def _keys(value, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name}: expected keys {sorted(expected)}")


def load_trait_data(path: str | Path, *, sha256: str) -> dict:
    """Read one upstream JSON, checking its exact bytes and required fields."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    data = json.loads(raw)
    _keys(data, {"questions", "instruction", "eval_prompt"}, str(path))
    questions = data["questions"]
    if not isinstance(questions, list) or not questions or any(
        not isinstance(q, str) or not q.strip() for q in questions
    ):
        raise ValueError("questions must be a nonempty list of nonempty strings.")
    if len(set(questions)) != len(questions):
        raise ValueError("Duplicate questions in source data.")
    instructions = data["instruction"]
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("instruction must be a nonempty list of pos/neg pairs.")
    for pair in instructions:
        _keys(pair, {"pos", "neg"}, "instruction pair")
        if any(not isinstance(s, str) or not s.strip() for s in pair.values()):
            raise ValueError("Instructions must be nonempty strings.")
    if not isinstance(data["eval_prompt"], str) or not data["eval_prompt"].strip():
        raise ValueError("eval_prompt must be a nonempty string.")
    return data


def load_persona_data(config_path: str | Path) -> dict:
    """Load the data config and both source files; validate the fixed split.

    Paths are relative to the YAML file, not the current working directory.
    Question IDs are zero-based positions in the corresponding source JSON.
    """
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _keys(config, {"trait", "source", "files", "split", "repeats"}, "data config")
    trait = config["trait"]
    if not isinstance(trait, str) or not re.fullmatch(r"[a-z][a-z_]*", trait):
        raise ValueError("trait must be a lowercase name, e.g. sycophantic.")
    _keys(config["source"], {"repository", "revision"}, "source")
    source = config["source"]
    if source["repository"] != "https://github.com/safety-research/persona_vectors":
        raise ValueError("Expected the official Persona Vectors repository.")
    if not isinstance(source["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", source["revision"]
    ):
        raise ValueError("source.revision must be a full commit SHA.")
    _keys(config["repeats"], {"extract", "eval"}, "repeats")
    if any(type(n) is not int or n < 1 for n in config["repeats"].values()):
        raise ValueError("repeats must be positive integers.")

    _keys(config["files"], {"extract", "eval"}, "files")
    bundle = {"config": config}
    for name, spec in config["files"].items():
        _keys(spec, {"path", "sha256"}, f"files.{name}")
        if not isinstance(spec["path"], str) or not spec["path"].strip():
            raise ValueError("File paths must be nonempty strings.")
        if not isinstance(spec["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", spec["sha256"]
        ):
            raise ValueError("File SHA-256 must contain 64 hexadecimal characters.")
        bundle[name] = load_trait_data(
            config_path.parent / spec["path"], sha256=spec["sha256"]
        )

    extract_questions = bundle["extract"]["questions"]
    eval_questions = bundle["eval"]["questions"]
    if set(extract_questions) & set(eval_questions):
        raise ValueError("Extraction and evaluation questions must not overlap.")
    _keys(config["split"], {"dev", "test"}, "split")
    for name, ids in config["split"].items():
        if not isinstance(ids, list) or not ids or any(
            type(i) is not int or not 0 <= i < len(eval_questions) for i in ids
        ):
            raise ValueError(f"split.{name}: expected nonempty valid question indices.")
        if len(set(ids)) != len(ids):
            raise ValueError(f"split.{name}: duplicate indices.")
    dev, test = (set(config["split"][name]) for name in ("dev", "test"))
    if dev & test or dev | test != set(range(len(eval_questions))):
        raise ValueError("dev/test must be disjoint and cover all evaluation questions.")
    return bundle


def build_extraction_jobs(data: dict) -> list[dict]:
    """Expand questions x instructions x repeats x {pos, neg} in that order.

    Each adjacent pos/neg pair shares a pair_id. Pair IDs join records later;
    they do not imply identical random draws during generation.
    """
    config = data["config"]
    trait = config["trait"]
    article = "an" if trait[0] in "aeiou" else "a"
    prefixes = {
        "pos": f"You are {article} {trait} assistant.",
        "neg": "You are a helpful assistant.",
    }
    jobs = []
    for question_id, question in enumerate(data["extract"]["questions"]):
        for instruction_id, pair in enumerate(data["extract"]["instruction"]):
            for repeat in range(config["repeats"]["extract"]):
                pair_id = f"{trait}_extract_q{question_id}_i{instruction_id}_r{repeat}"
                for polarity in ("pos", "neg"):
                    jobs.append({
                        "id": f"{pair_id}_{polarity}", "pair_id": pair_id,
                        "trait": trait, "split": "extract",
                        "question_id": question_id, "instruction_id": instruction_id,
                        "repeat": repeat, "polarity": polarity, "question": question,
                        "system": f"{prefixes[polarity]} {pair[polarity]}",
                    })
    return jobs


def build_eval_jobs(data: dict, *, split: str) -> list[dict]:
    """Build dev OR test jobs, preserving config order; no explicit system prompt."""
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test.")
    config = data["config"]
    trait = config["trait"]
    return [
        {
            "id": f"{trait}_{split}_q{question_id}_r{repeat}",
            "trait": trait, "split": split, "question_id": question_id,
            "repeat": repeat, "question": data["eval"]["questions"][question_id],
            "system": None,
        }
        for question_id in config["split"][split]
        for repeat in range(config["repeats"]["eval"])
    ]
