import unittest

from llm_race_lab.metrics import percentile, summarize


class MetricsTest(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3], 0.5), 2)
        self.assertEqual(percentile([], 0.95), None)

    def test_summary(self):
        summary = summarize(
            [
                {
                    "status": "ok",
                    "latency_ms": 100,
                    "ttft_ms": 10,
                    "tpot_ms": 5,
                    "output_tokens": 20,
                    "started_at_epoch": 1.0,
                    "finished_at_epoch": 2.0,
                },
                {
                    "status": "error",
                    "latency_ms": 200,
                    "ttft_ms": None,
                    "output_tokens": 0,
                    "started_at_epoch": 1.5,
                    "finished_at_epoch": 3.0,
                },
            ]
        )
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["wall_duration_s"], 2.0)


if __name__ == "__main__":
    unittest.main()
