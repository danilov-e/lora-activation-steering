import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import BadRequestError, OpenAI

from src.artifacts import write_bytes, write_json
from src.judge import judge, load_ratings
from src.workflow import default_config


class JudgeTest(unittest.TestCase):
    def test_default_request_and_resume(self):
        config = default_config()["judge"]
        self.assertEqual((config["model"], config["reasoning"]), ("gpt-6.1-sol", "xhigh"))
        self.assertEqual(config["batch_size"], 5)
        requests = []

        def respond(request):
            self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "gpt-6.1-sol")
            self.assertEqual(body["reasoning"], {"effort": "xhigh"})
            self.assertFalse(body["store"])
            self.assertEqual(body["max_output_tokens"], config["max_output_tokens"])
            items = json.loads(body["input"])["items"]
            self.assertLessEqual(len(items), 5)
            self.assertEqual(len({item["question"] for item in items}), len(items))
            for item in items:
                self.assertEqual(set(item), {"id", "question", "answer"})
            requests.append(items)
            if len(requests) == 2:
                return httpx.Response(400, json={"error": {"message": "Test interruption"}})

            ratings = [{"id": item["id"], "evil": 15, "coherence": 95} for item in items]
            return httpx.Response(200, json={
                "id": f"resp_test_{len(requests)}",
                "object": "response",
                "created_at": 1,
                "model": body["model"],
                "status": "completed",
                "output": [{
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": json.dumps({"ratings": ratings}),
                        "annotations": [],
                    }],
                }],
            })

        def client(**kwargs):
            return OpenAI(
                api_key="test-key",
                http_client=httpx.Client(transport=httpx.MockTransport(respond)),
                **kwargs,
            )

        with tempfile.TemporaryDirectory() as temp, patch("src.judge.OpenAI", client):
            output = Path(temp)
            rows = [
                {"id": f"id_{question}_{repeat}", "question_id": f"q_{question}",
                 "question": f"Question {question}?", "answer": "Example answer"}
                for repeat in range(2) for question in range(5)
            ]
            write_json(output / "manifest.json", {"expected": len(rows)})
            write_bytes(output / "answers.jsonl", "".join(json.dumps(row) + "\n" for row in rows).encode())

            with self.assertRaises(BadRequestError):
                judge(output, "evil")
            with self.assertRaisesRegex(ValueError, "batches are still missing"):
                load_ratings(output, "evil")
            judge(output, "evil", **config)
            self.assertEqual(requests[1], requests[2])

            judge(output, "evil", **config)
            self.assertEqual(len(requests), 3)
            _, ratings = load_ratings(output, "evil")
            self.assertEqual({rating["id"] for rating in ratings}, {row["id"] for row in rows})

            with self.assertRaisesRegex(ValueError, "different answers or judge settings"):
                judge(output, "evil", reasoning="high")
