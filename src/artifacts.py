"""I/O for files saved by the experiment."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        name = Path(temporary.name)
        temporary.write(data)
    try:
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    write_bytes(path, (text + "\n").encode("utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
