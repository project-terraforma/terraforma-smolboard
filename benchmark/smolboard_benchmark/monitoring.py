"""Best-effort runtime and memory measurements."""

from __future__ import annotations

import subprocess
import threading
from typing import Any

import psutil


def query_gpu_memory_mib() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


class ResourceMonitor:
    def __init__(self, process: subprocess.Popen[str] | None = None) -> None:
        self.process = process
        self.baseline_gpu_mib = query_gpu_memory_mib()
        self.peak_gpu_mib = self.baseline_gpu_mib
        self.baseline_system_ram_mib = psutil.virtual_memory().used / (1024**2)
        self.peak_system_ram_mib = self.baseline_system_ram_mib
        self.peak_server_rss_mib: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=5)
        gpu_delta = None
        if self.peak_gpu_mib is not None and self.baseline_gpu_mib is not None:
            gpu_delta = max(0.0, self.peak_gpu_mib - self.baseline_gpu_mib)
        return {
            "gpu_memory_baseline_mib": self.baseline_gpu_mib,
            "gpu_memory_peak_total_mib": self.peak_gpu_mib,
            "gpu_memory_peak_delta_mib": gpu_delta,
            "system_ram_baseline_used_mib": self.baseline_system_ram_mib,
            "system_ram_peak_used_mib": self.peak_system_ram_mib,
            "server_peak_rss_mib": self.peak_server_rss_mib,
            "resource_measurement_note": (
                "GPU values are whole-device nvidia-smi samples and include unrelated desktop use; "
                "RAM values are sampled, not profiler-grade measurements."
            ),
        }

    def _sample(self) -> None:
        while not self._stop.wait(0.5):
            gpu = query_gpu_memory_mib()
            if gpu is not None:
                self.peak_gpu_mib = max(self.peak_gpu_mib or gpu, gpu)
            self.peak_system_ram_mib = max(self.peak_system_ram_mib, psutil.virtual_memory().used / (1024**2))
            if self.process is not None and self.process.poll() is None:
                try:
                    proc = psutil.Process(self.process.pid)
                    rss = proc.memory_info().rss + sum(child.memory_info().rss for child in proc.children(recursive=True))
                    self.peak_server_rss_mib = max(self.peak_server_rss_mib or 0.0, rss / (1024**2))
                except (psutil.Error, OSError):
                    pass
