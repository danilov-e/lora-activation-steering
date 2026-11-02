"""Prepare pinned Persona Vectors SFT train/validation files. Run from repo root."""

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path
from urllib.request import urlopen

from src.data import load_persona_data
from src.sft_data import build_sft_splits, load_sft_config, load_sft_source


def download_archive(path: Path, source: dict) -> None:
    """Cache one verified archive; existing cache bytes are validated by the loader."""
    if path.exists():
        return
    url = (f"https://raw.githubusercontent.com/safety-research/persona_vectors/"
           f"{source['revision']}/dataset.zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != source["archive_sha256"]:
            raise ValueError("Downloaded archive SHA-256 mismatch.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft_sycophantic.yaml"))
    parser.add_argument("--archive", type=Path, default=Path("outputs/cache/persona_vectors_dataset.zip"))
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config_path = args.config.resolve()
    config = load_sft_config(config_path)
    persona_path = config_path.parent / config["persona_config"]
    persona = load_persona_data(persona_path)
    if persona["config"]["trait"] != config["trait"]:
        raise ValueError("SFT and Persona Vectors traits differ.")
    download_archive(args.archive, config["source"])
    rows = load_sft_source(args.archive, config["source"])
    forbidden = persona["extract"]["questions"] + persona["eval"]["questions"]
    splits, report = build_sft_splits(rows, forbidden, config["split"])
    payloads = {
        name: "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values).encode("utf-8")
        for name, values in splits.items()
    }
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "config": config, "persona_data_config": persona["config"], "report": report,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "persona_config_sha256": hashlib.sha256(persona_path.read_bytes()).hexdigest(),
        "files": {f"{name}.jsonl": {"n_records": len(splits[name]),
                  "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in payloads.items()},
        "code_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                        for p in ("src/sft_data.py", "src/data.py", "scripts/prepare_sft_data.py")},
        "python": platform.python_version(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        (args.output / f"{name}.jsonl").write_bytes(raw)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
