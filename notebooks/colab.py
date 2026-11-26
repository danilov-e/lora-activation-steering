"""Common setup for the three Colab notebooks."""
import json
import os
import subprocess
import sys
from getpass import getpass
from pathlib import Path


def _git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()


def setup(source, runs_dir, run_name, trait, ref="main", *, resume=False):
    from google.colab import drive

    drive.mount("/content/drive")
    runs = Path(runs_dir)
    manifest = runs / run_name / trait / "run.json"
    if manifest.exists():
        ref = json.loads(manifest.read_text(encoding="utf-8")).get("git_commit")
        if not ref:
            raise ValueError("This is a historical run. Open its archived notebook or use a new RUN_NAME.")
    elif resume:
        raise FileNotFoundError(f"{manifest}: finish notebook 01 on this Google Drive first.")

    _git(source, "fetch", "--quiet", "origin", ref)
    commit = _git(source, "rev-parse", "FETCH_HEAD^{commit}")
    project = source.parent / commit
    if not project.exists():
        project.parent.mkdir(parents=True, exist_ok=True)
        _git(source, "worktree", "add", "--quiet", "--detach", str(project), commit)
    elif _git(project, "rev-parse", "HEAD") != commit or _git(project, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"{project}: expected a clean checkout of {commit}. Start a fresh Colab session.")

    loaded = sys.modules.get("src")
    if loaded is not None and Path(loaded.__file__).resolve().parent != project / "src":
        raise RuntimeError("A different project version is already loaded. Restart this Colab session.")

    subprocess.run(
        [sys.executable, "-m", "pip", "-q", "install", "-r", str(project / "requirements.txt")],
        check=True,
    )
    sys.path.insert(0, str(project))
    print("Commit:", commit)
    print("Output directory:", runs)
    return project, runs, commit


def api_key():
    from google.colab import userdata

    if os.environ.get("OPENAI_API_KEY"):
        return
    try:
        key = userdata.get("OPENAI_API_KEY")
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
        key = None
    key = key or getpass("OpenAI API key: ")
    if not key.strip():
        raise ValueError("Set OPENAI_API_KEY in Colab Secrets or enter an API key.")
    os.environ["OPENAI_API_KEY"] = key.strip()
