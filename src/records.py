"""Read and validate the JSONL records used by the generation commands."""

import hashlib
import json
from pathlib import Path


def user_messages(question: str, system: str | None = None) -> list[dict]:
    messages = [] if system is None else [{"role": "system", "content": system}]
    return [*messages, {"role": "user", "content": question}]


def generation_messages(jobs: list[dict]) -> list[list[dict]]:
    """Validate job IDs and return the chat input for each job in order."""
    seen, conversations = set(), []
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Every job must be a JSON object.")
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("Every job needs a nonempty string id.")
        if job_id in seen:
            raise ValueError(f"Duplicate job ids: {job_id!r}.")
        seen.add(job_id)

        if "messages" not in job:
            # Single-turn inputs are still used by the generation smoke entry point.
            question, system = job.get("question"), job.get("system")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{job_id}: expected messages or a nonempty question.")
            if system is not None and not isinstance(system, str):
                raise ValueError(f"{job_id}: system must be a string or None.")
            conversations.append(user_messages(question, system))
            continue

        group_id = job.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"{job_id}: chat records need a nonempty group_id.")
        messages = job["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{job_id}: messages must be a nonempty list.")
        expected_role = "user"
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError(f"{job_id}: messages must contain text content.")
            role = message.get("role")
            if index == 0 and role == "system":
                continue
            if role != expected_role or not message["content"].strip():
                raise ValueError(f"{job_id}: expected a nonempty {expected_role} message.")
            expected_role = "assistant" if role == "user" else "user"
        if expected_role != "assistant":
            raise ValueError(f"{job_id}: a generation prompt must end with a user message.")
        conversations.append(messages)
    return conversations


def read_jsonl(path: Path, *, raw: bytes | None = None) -> list[dict]:
    rows = []
    content = path.read_bytes() if raw is None else raw
    for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{number}; saved data was not changed.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {path}:{number}.")
        rows.append(row)
    return rows


def load_prepared_jobs(path: Path) -> tuple[list[dict], str, dict | None]:
    """Read jobs and verify a preparation manifest when one accompanies them."""
    raw = path.read_bytes()
    jobs = read_jsonl(path, raw=raw)
    if not jobs:
        raise ValueError("The jobs file is empty.")
    generation_messages(jobs)
    digest = hashlib.sha256(raw).hexdigest()
    manifest_path = path.with_name("manifest.json")
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if not isinstance(manifest, dict) or manifest.get("jobs_sha256") != digest:
            raise ValueError("Input jobs do not match their preparation manifest.")
        if manifest.get("n_jobs") != len(jobs):
            raise ValueError("Input job count does not match the preparation manifest.")
    return jobs, digest, manifest