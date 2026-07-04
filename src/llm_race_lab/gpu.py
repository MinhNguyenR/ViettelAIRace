from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path


GPU_QUERY = [
    "timestamp",
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
]


class NvidiaSmiSampler:
    def __init__(self, output: Path, interval_s: float = 1.0) -> None:
        self.output = output
        self.interval_s = max(0.2, interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def __enter__(self) -> "NvidiaSmiSampler":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 2)

    def _sample(self) -> list[str]:
        cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_QUERY)}",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "nvidia-smi failed")
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    def _run(self) -> None:
        with self.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_epoch", *GPU_QUERY])
            while not self._stop.is_set():
                try:
                    sample_epoch = f"{time.time():.6f}"
                    for line in self._sample():
                        writer.writerow([sample_epoch, *[part.strip() for part in line.split(",")]])
                    f.flush()
                except Exception as exc:  # noqa: BLE001 - sampler must not kill benchmark.
                    self.error = str(exc)
                    error_path = self.output.with_suffix(".error.txt")
                    error_path.write_text(self.error, encoding="utf-8")
                    return
                self._stop.wait(self.interval_s)
