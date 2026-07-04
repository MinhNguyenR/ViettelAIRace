import json
import tempfile
import unittest
from pathlib import Path

from llm_race_lab.workload import generate_records, generate_workload


class WorkloadTest(unittest.TestCase):
    def test_generate_workload_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workload.jsonl"
            generate_workload(path, requests=5, seed=1)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertIn("prompt", rows[0])
            self.assertIn("max_tokens", rows[0])

    def test_generate_fixed_classes(self):
        rows = generate_records(
            requests=3,
            seed=7,
            prompt_class="long",
            decode_class="short",
            pattern="burst",
            case_id="case",
        )
        self.assertEqual({row["prompt_class"] for row in rows}, {"long"})
        self.assertEqual({row["decode_class"] for row in rows}, {"short"})
        self.assertEqual({row["request_pattern"] for row in rows}, {"burst"})
        self.assertTrue(rows[0]["request_id"].startswith("case-"))


if __name__ == "__main__":
    unittest.main()
