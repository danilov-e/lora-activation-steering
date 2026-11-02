"""Prepare pinned Persona Vectors dialogues for supervised fine-tuning.

The normal variant is a reference dataset, not a verified anti-sycophancy set.
No model loading, tokenization or training happens here.
"""

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import yaml


def _keys(value, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name}: expected keys {sorted(expected)}")


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_sft_config(path: str | Path) -> dict:
    """Read a strict config. persona_config is relative to this YAML file."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _keys(config, {"trait", "persona_config", "source", "split"}, "SFT data config")
    if config["trait"] != "sycophantic":
        raise ValueError("This loader supports the sycophantic trait.")
    if not isinstance(config["persona_config"], str) or not config["persona_config"].strip():
        raise ValueError("persona_config must be a nonempty path.")
    source = config["source"]
    _keys(source, {"repository", "revision", "archive_sha256", "member", "member_sha256"}, "source")
    if source["repository"] != "https://github.com/safety-research/persona_vectors":
        raise ValueError("Expected the official Persona Vectors repository.")
    for key, length in (("revision", 40), ("archive_sha256", 64), ("member_sha256", 64)):
        if not isinstance(source[key], str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", source[key]):
            raise ValueError(f"source.{key} must contain {length} hexadecimal characters.")
    if not isinstance(source["member"], str) or not re.fullmatch(
        r"dataset/sycophancy/(normal|misaligned_[12])\.jsonl", source["member"]
    ):
        raise ValueError("Expected a Persona Vectors sycophancy JSONL member.")
    split = config["split"]
    _keys(split, {"seed", "validation_fraction", "train_size", "validation_size"}, "split")
    if type(split["seed"]) is not int or not 0 <= split["seed"] < 2**32:
        raise ValueError("split.seed must be an integer in [0, 2**32).")
    for key in ("train_size", "validation_size"):
        if type(split[key]) is not int or split[key] < 1:
            raise ValueError(f"split.{key} must be a positive integer.")
    fraction = split["validation_fraction"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    return config


def _question_key(prompt: str) -> tuple[str, str]:
    """Remove biographies in the pinned NLP, politics and factual prompt formats.

    Keep choices, sorting them so swapping A/B cannot cross the split boundary.
    This groups matching source questions; it is not semantic deduplication.
    """
    matches = list(re.finditer(r"\s+\(A\)", prompt))
    if not matches:
        raise ValueError("Source prompt has no answer choices.")
    match = matches[-1]
    head = prompt[:match.start()].strip()
    choices = re.split(r"\s+\([A-Z]\)\s*", prompt[match.start():])[1:]
    if len(choices) < 2 or any(not choice.strip() for choice in choices):
        raise ValueError("Source prompt needs at least two nonempty answer choices.")
    marker = "Do you agree or disagree with the following claim"
    if "Question:" in head:
        domain = "factual"
        task = head.rsplit("Question:", 1)[1].split("Answers:", 1)[0]
    elif marker in head:
        domain = "nlp"
        task = marker + head.split(marker, 1)[1].split("\nChoices:", 1)[0]
    else:
        domain = "politics"
        task = re.split(r"(?<=[.!?])\s+(?=[A-Z])", head)[-1]
    if not task.strip():
        raise ValueError("Source question is empty.")
    canonical = json.dumps([domain, _normalize(task), sorted(_normalize(c) for c in choices)],
                           ensure_ascii=False)
    return domain, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_sft_source(archive_path: str | Path, source: dict) -> list[dict]:
    """Check archive/member bytes and read original user/assistant dialogues."""
    archive_path = Path(archive_path)
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != source["archive_sha256"]:
        raise ValueError("Archive SHA-256 mismatch.")
    with ZipFile(archive_path) as archive:
        if archive.namelist().count(source["member"]) != 1:
            raise ValueError("Expected exactly one configured archive member.")
        raw = archive.read(source["member"])
    if hashlib.sha256(raw).hexdigest() != source["member_sha256"]:
        raise ValueError("Member SHA-256 mismatch.")
    rows, prompts = [], set()
    for source_row, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        _keys(row, {"messages"}, "source record")
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("Each record must have one user and one assistant message.")
        for message, role in zip(messages, ("user", "assistant")):
            _keys(message, {"role", "content"}, "message")
            if (message["role"] != role or not isinstance(message["content"], str)
                    or not message["content"].strip()):
                raise ValueError("Expected nonempty user/assistant messages in that order.")
        prompt = messages[0]["content"]
        normalized = _normalize(prompt)
        if normalized in prompts:
            raise ValueError("Duplicate normalized user prompt in source data.")
        prompts.add(normalized)
        domain, group_id = _question_key(prompt)
        rows.append({"id": f"sycophancy_sft_{source_row}", "source_row": source_row,
                     "domain": domain, "group_id": group_id, "messages": messages})
    if not rows:
        raise ValueError("Source data is empty.")
    return rows


def build_sft_splits(rows: list[dict], forbidden_questions: list[str], settings: dict) -> tuple[dict, dict]:
    """Hold out question groups within each domain, then select fixed-size subsets.

    SHA-256 ordering uses a local seed without changing process RNGs. Exact
    evaluation question text embedded in a source prompt excludes its whole group.
    Input records and assistant responses are preserved.
    """
    blocked = [_normalize(q) for q in forbidden_questions]
    excluded_groups = {r["group_id"] for r in rows
                       if any(q in _normalize(r["messages"][0]["content"]) for q in blocked)}
    eligible = [r for r in rows if r["group_id"] not in excluded_groups]
    groups = defaultdict(set)
    for row in eligible:
        groups[row["domain"]].add(row["group_id"])

    def order(value: str) -> str:
        return hashlib.sha256(f"{settings['seed']}:{value}".encode("utf-8")).hexdigest()

    validation_groups = set()
    for domain, ids in groups.items():
        if len(ids) < 2:
            raise ValueError(f"Need at least two eligible question groups in {domain}.")
        n = min(len(ids) - 1, max(1, math.ceil(len(ids) * settings["validation_fraction"])))
        validation_groups.update(sorted(ids, key=order)[:n])
    pools = {
        "train": [r for r in eligible if r["group_id"] not in validation_groups],
        "validation": [r for r in eligible if r["group_id"] in validation_groups],
    }
    splits = {}
    for name, pool in pools.items():
        size = settings[f"{name}_size"]
        if len(pool) < size:
            raise ValueError(f"Not enough {name} records: requested {size}, available {len(pool)}.")
        splits[name] = [{**r, "split": name} for r in sorted(pool, key=lambda r: order(r["id"]))[:size]]
    report = {
        "source_records": len(rows), "source_groups": len({r["group_id"] for r in rows}),
        "excluded_records": len(rows) - len(eligible), "excluded_groups": len(excluded_groups),
        "domains": {
            name: {"pool_records": len(pool), "selected_records": len(splits[name]),
                   "selected_groups": len({r["group_id"] for r in splits[name]}),
                   "selected_domains": {d: sum(r["domain"] == d for r in splits[name])
                                        for d in sorted(groups)}}
            for name, pool in pools.items()
        },
        "grouping": "source_question_and_sorted_choices_without_biography",
        "ordering": "sha256(seed:group_id), sha256(seed:record_id)",
    }
    return splits, report
