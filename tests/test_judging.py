import asyncio
import copy
import json
import math
import shutil
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from openai import AsyncOpenAI

from scripts import filter_pairs as filter_cli
from scripts import judge_answers as judge_cli
from src.data import build_extraction_jobs, load_persona_data
from src.filtering import filter_pairs
from src.judge import aggregate_score, judge_answer, load_judge_config, load_scores

ROOT = Path(__file__).resolve().parents[1]


def completion(number=80):
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
        "model": "gpt-4.1-mini-2025-04-14", "system_fingerprint": "test-fingerprint",
        "choices": [{"index": 0, "finish_reason": "length",
                     "message": {"role": "assistant", "content": "0"},
                     "logprobs": {"content": [{"token": "0", "logprob": -100,
                                               "top_logprobs": [{"token": str(number), "logprob": 0}]}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


@pytest.mark.parametrize("top,expected,mass", [
    ([{"token": "0", "logprob": math.log(0.2)},
      {"token": "100", "logprob": math.log(0.4)},
      {"token": "REFUSAL", "logprob": math.log(0.4)}], 100 * 0.4 / 0.6, 0.6),
    ([{"token": "50", "logprob": math.log(0.25)}], 50, 0.25),
    ([{"token": "50", "logprob": math.log(0.24)}], None, 0.24),
    ([{"token": "101", "logprob": 0}], None, 0),
    ([{"token": " 80", "logprob": 0}], 80, 1),
])
def test_author_probability_aggregation(top, expected, mass):
    score, numeric_mass = aggregate_score(top)
    assert numeric_mass == pytest.approx(mass)
    assert score is None if expected is None else score == pytest.approx(expected)


def test_real_sdk_request_and_raw_response():
    requests = []
    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=completion(80))

    async def run():
        async with AsyncOpenAI(api_key="test-key", base_url="https://judge.test/v1",
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            return await judge_answer(client, "judge-model", "Q: {question}\nA: {answer}",
                                      question="question", answer="answer with {braces}")
    result = asyncio.run(run())
    assert result["score"] == 80
    assert result["numeric_mass"] == 1
    assert result["response"]["choices"][0]["message"]["content"] == "0"
    assert result["response"]["usage"]["total_tokens"] == 11
    assert requests == [{
        "model": "judge-model", "messages": [{"role": "user", "content": "Q: question\nA: answer with {braces}"}],
        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20, "seed": 0,
    }]


@pytest.fixture
def prepared_judging(tmp_path, monkeypatch):
    data = load_persona_data(ROOT / "configs/persona_sycophantic.yaml")
    answers = build_extraction_jobs(data)[:4]
    for i, row in enumerate(answers):
        row["answer"] = f"TEST_ANSWER_{i}"
        row["generated_token_ids"] = [100 + i, 151645]
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text("".join(json.dumps(r) + "\n" for r in answers), encoding="utf-8")
    config = yaml.safe_load((ROOT / "configs/judge.yaml").read_text())
    config.update(api_key_env="JUDGE_TEST_KEY", max_retries=0, concurrency=2)
    config["coherence_prompt"]["path"] = str(ROOT / "data/persona_vectors/coherence_prompt.txt")
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("JUDGE_TEST_KEY", "not-a-real-secret")
    return answers_path, config_path, answers


@pytest.fixture
def backend(monkeypatch):
    state = {"requests": [], "active": 0, "peak": 0, "fail_at": None,
             "missing_logprobs": False, "no_pairs": False, "unscorable": False}

    async def handle(request):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        body = json.loads(request.content)
        state["requests"].append(body)
        fail = len(state["requests"]) == state["fail_at"]
        try:
            await asyncio.sleep(0.005)
            if fail:
                return httpx.Response(503, json={"error": {"message": "simulated failure", "type": "server_error"}})
            prompt = body["messages"][0]["content"]
            coherence = prompt.startswith("I am evaluating a different language model.")
            is_neg = "TEST_ANSWER_1" in prompt or "TEST_ANSWER_3" in prompt
            number = (40 if "TEST_ANSWER_3" in prompt else 100) if coherence else (20 if is_neg else 80)
            if state["no_pairs"]:
                number = 100 if coherence else 80
            response = completion(number)
            if state["unscorable"]:
                response["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = [
                    {"token": "REFUSAL", "logprob": 0}
                ]
            if state["missing_logprobs"]:
                response["choices"][0]["logprobs"] = None
            return httpx.Response(200, json=response)
        finally:
            state["active"] -= 1

    def factory(**kwargs):
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(judge_cli, "AsyncOpenAI", factory)
    return state


def run_judge(monkeypatch, prepared, output, *extra):
    answers_path, config_path, _ = prepared
    monkeypatch.setattr(sys, "argv", ["judge_answers", "--config", str(config_path),
                                    "--data-config", str(ROOT / "configs/persona_sycophantic.yaml"),
                                    "--answers", str(answers_path), "--output", str(output), *extra])
    judge_cli.main()


def test_smoke_resume_and_filter_end_to_end(tmp_path, monkeypatch, prepared_judging, backend):
    answers_path, _, answers = prepared_judging
    output = tmp_path / "judged"
    run_judge(monkeypatch, prepared_judging, output, "--limit", "2")
    assert len(backend["requests"]) == 4
    assert json.loads((output / "summary.json").read_text())["status"] == "partial"
    with pytest.raises(FileExistsError):
        run_judge(monkeypatch, prepared_judging, output)
    assert len(backend["requests"]) == 4
    selected = tmp_path / "selected"
    monkeypatch.setattr(sys, "argv", ["filter_pairs", "--answers", str(answers_path),
                                    "--judgments", str(output), "--output", str(selected)])
    with pytest.raises(ValueError, match="Missing judgment"):
        filter_cli.main()
    assert not selected.exists()

    run_judge(monkeypatch, prepared_judging, output, "--resume")
    assert len(backend["requests"]) == 8
    assert backend["peak"] <= 2
    run_judge(monkeypatch, prepared_judging, output, "--resume")
    assert len(backend["requests"]) == 8
    assert json.loads((output / "summary.json").read_text())["status"] == "complete"
    assert "not-a-real-secret" not in (output / "manifest.json").read_text()
    for request in backend["requests"]:
        assert len(request["messages"]) == 1 and request["messages"][0]["role"] == "user"
        assert "You are a sycophantic assistant." not in request["messages"][0]["content"]

    monkeypatch.setattr(sys, "argv", ["filter_pairs", "--answers", str(answers_path),
                                    "--judgments", str(output), "--output", str(selected)])
    filter_cli.main()
    retained = [json.loads(l) for l in (selected / "selected_answers.jsonl").read_text().splitlines()]
    assert retained == answers[:2]
    summary = json.loads((selected / "summary.json").read_text())
    assert summary["kept_pairs"] == 1 and summary["rejected_pairs"] == 1
    assert summary["coverage"]["question_id"] == {"0": 1}
    assert summary["failed_checks"] == {"neg_coherence": 1}
    reports = [json.loads(l) for l in (selected / "pairs.jsonl").read_text().splitlines()]
    assert reports[0]["kept"] and not reports[1]["kept"]


def test_api_failure_preserves_cache_and_resume_retries_only_missing(tmp_path, monkeypatch, prepared_judging, backend):
    _, config_path, _ = prepared_judging
    config = yaml.safe_load(config_path.read_text())
    config["concurrency"] = 1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    backend["fail_at"] = 3
    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="simulated failure"):
        run_judge(monkeypatch, prepared_judging, output)
    saved = (output / "judgments.jsonl").read_bytes()
    assert len(saved.splitlines()) == 2
    backend["fail_at"] = None
    run_judge(monkeypatch, prepared_judging, output, "--resume")
    assert (output / "judgments.jsonl").read_bytes().startswith(saved)
    assert len(backend["requests"]) == 9


@pytest.mark.parametrize("change", ["answers", "config", "cache"])
def test_resume_rejects_changes_before_api(tmp_path, monkeypatch, prepared_judging, backend, change):
    answers_path, config_path, _ = prepared_judging
    output = tmp_path / "run"
    run_judge(monkeypatch, prepared_judging, output, "--limit", "2")
    path = output / "judgments.jsonl"
    if change == "answers":
        answers_path.write_bytes(answers_path.read_bytes() + b"\n")
    elif change == "config":
        config = yaml.safe_load(config_path.read_text())
        config["model"] = "other-judge"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b'{"id":')
    before = path.read_bytes()
    with pytest.raises(ValueError):
        run_judge(monkeypatch, prepared_judging, output, "--resume")
    assert path.read_bytes() == before and len(backend["requests"]) == 4


def test_missing_logprobs_stops_instead_of_using_completion_text(tmp_path, monkeypatch, prepared_judging, backend):
    backend["missing_logprobs"] = True
    output = tmp_path / "unsupported"
    with pytest.raises(RuntimeError, match="top_logprobs"):
        run_judge(monkeypatch, prepared_judging, output, "--limit", "2")
    assert not (output / "judgments.jsonl").read_bytes()
    assert not (output / "summary.json").exists()


def test_prompt_checksum_is_checked(tmp_path):
    config = yaml.safe_load((ROOT / "configs/judge.yaml").read_text())
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("different prompt", encoding="utf-8")
    config["coherence_prompt"]["path"] = str(prompt)
    path = tmp_path / "judge.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_judge_config(path)


@pytest.mark.parametrize("preset", [False, True])
def test_env_file_loading_and_environment_precedence(tmp_path, monkeypatch, prepared_judging, backend, preset):
    repo = tmp_path / "repo"
    for name in ("scripts/judge_answers.py", "src/judge.py", "src/data.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    (repo / ".env").write_text(
        "OPENAI_API_KEY=dotenv-test-secret\nOPENAI_BASE_URL=https://dotenv.test/v1\n", encoding="utf-8"
    )
    monkeypatch.setattr(judge_cli, "__file__", str(repo / "scripts/judge_answers.py"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    if preset:
        monkeypatch.setenv("OPENAI_API_KEY", "environment-test-secret")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.test/v1")
    config_path = prepared_judging[1]
    config = yaml.safe_load(config_path.read_text())
    config["api_key_env"] = "OPENAI_API_KEY"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    factory = judge_cli.AsyncOpenAI
    calls = []
    def record_client(**kwargs):
        calls.append(kwargs)
        return factory(**kwargs)
    monkeypatch.setattr(judge_cli, "AsyncOpenAI", record_client)
    output = tmp_path / "env_run"
    run_judge(monkeypatch, prepared_judging, output, "--limit", "2")
    expected_key = "environment-test-secret" if preset else "dotenv-test-secret"
    expected_url = "https://environment.test/v1" if preset else "https://dotenv.test/v1"
    assert calls[0]["api_key"] == expected_key
    assert calls[0]["base_url"] == expected_url
    manifest_text = (output / "manifest.json").read_text()
    assert "dotenv-test-secret" not in manifest_text
    assert "environment-test-secret" not in manifest_text
    assert json.loads(manifest_text)["spec"]["judge_config"]["base_url"] == expected_url


@pytest.mark.parametrize("duplicate", [False, True])
def test_invalid_scores_are_rejected(tmp_path, duplicate):
    record = {"id": "a", "metric": "coherence", "score": 80,
              "numeric_mass": 1, "status": "ok"}
    records = [record, record] if duplicate else [{**record, "score": float("nan")}]
    path = tmp_path / "scores.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    with pytest.raises(ValueError):
        load_scores(path, answer_ids={"a"}, metrics={"coherence"})


@pytest.mark.parametrize("values,kept", [
    ((50, 49.999, 50, 50), True),
    ((50, 50, 100, 100), False),
    ((49.999, 0, 100, 100), False),
    ((100, 0, 49.999, 100), False),
    ((100, 0, 100, None), False),
])
def test_exact_author_thresholds_and_order_independent_pairing(prepared_judging, values, kept):
    rows = copy.deepcopy(prepared_judging[2][:2])
    trait = "sycophantic"
    scores = {(rows[0]["id"], trait): {"score": values[0]},
              (rows[1]["id"], trait): {"score": values[1]},
              (rows[0]["id"], "coherence"): {"score": values[2]},
              (rows[1]["id"], "coherence"): {"score": values[3]}}
    reports, selected = filter_pairs(list(reversed(rows)), scores, trait=trait)
    assert reports[0]["kept"] is kept
    assert selected == list(reversed(rows)) if kept else selected == []
    with pytest.raises(ValueError, match="Incomplete pair"):
        filter_pairs(rows[:1], scores, trait=trait)
    with pytest.raises(ValueError, match="Duplicate answer ids"):
        filter_pairs(rows + rows[:1], scores, trait=trait)


@pytest.mark.parametrize("mode", ["no_pairs", "unscorable"])
def test_zero_pairs_saves_diagnostics_and_returns_nonzero(tmp_path, monkeypatch, prepared_judging, backend, mode):
    backend[mode] = True
    output, selected = tmp_path / "judged", tmp_path / "selected"
    run_judge(monkeypatch, prepared_judging, output)
    if mode == "unscorable":
        summary = json.loads((output / "summary.json").read_text())
        assert summary["unscorable"] == {"sycophantic": 4, "coherence": 4}
        assert summary["means"] == {"sycophantic": None, "coherence": None}
        run_judge(monkeypatch, prepared_judging, output, "--resume")
        assert len(backend["requests"]) == 8
    monkeypatch.setattr(sys, "argv", ["filter_pairs", "--answers", str(prepared_judging[0]),
                                    "--judgments", str(output), "--output", str(selected)])
    with pytest.raises(SystemExit) as exc:
        filter_cli.main()
    assert exc.value.code == 2
    assert (selected / "selected_answers.jsonl").read_bytes() == b""
    assert json.loads((selected / "summary.json").read_text())["kept_pairs"] == 0
