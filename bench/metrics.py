"""Timing percentiles and a background resource sampler."""
from __future__ import annotations

import threading
import time

import numpy as np
import psutil


def percentiles(latencies_ms: list[float]) -> dict:
    a = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "count": int(a.size),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
        "mean_ms": round(float(a.mean()), 3),
        "min_ms": round(float(a.min()), 3),
        "max_ms": round(float(a.max()), 3),
    }


class ResourceSampler:
    """Context manager: peak process RSS (MB) + mean system CPU% during a phase.

    Most meaningful for the in-process Edge cells, where the DB work happens inside
    this process. For server cells the engine runs elsewhere, so treat client-side
    RSS as informational only.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_mb = 0.0
        self._cpu_samples: list[float] = []

    def __enter__(self) -> "ResourceSampler":
        psutil.cpu_percent(None)  # prime the interval baseline
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_mb = max(self.peak_rss_mb, self.proc.memory_info().rss / 1e6)
            self._cpu_samples.append(psutil.cpu_percent(None))
            time.sleep(self.interval)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        cpu = float(np.mean(self._cpu_samples)) if self._cpu_samples else 0.0
        return {"peak_rss_mb": round(self.peak_rss_mb, 1), "mean_cpu_pct": round(cpu, 1)}
