"""Shared mixins for causal discovery and groups extraction base classes.

StandardizationMixin
    Provides ``initialize_data`` which stores a (optionally standardized)
    copy of the input array in ``self._data``.

MemoryMonitorMixin
    Provides ``measure_execution`` which wraps any callable with a
    wall-clock timer and a background memory-usage thread.
"""

import logging
import os
import threading
import time
from typing import Any, Callable, TypeVar

import numpy as np
import psutil

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

class StandardizationMixin:
    """Mixin that standardizes input data and stores it as ``self._data``.

    Subclasses should call ``self.initialize_data(data, standarize)``
    inside their ``__init__`` (typically via ``super().__init__(…)``).
    """

    def initialize_data(self, data: np.ndarray, standarize: bool = True) -> None:
        """Store *data* (optionally standardized) in ``self._data``.

        Standardization subtracts the mean and divides by the standard
        deviation per feature.  Features with zero variance are left as
        zeros so that no division-by-zero occurs.
        """
        if standarize:
            self._data = data - data.mean(axis=0)
            std = self._data.std(axis=0)
            if np.all(std != 0):
                self._data /= std
        else:
            self._data = data


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

class MemoryMonitorMixin:
    """Mixin that measures wall-clock time and peak RSS of a callable.

    Subclasses gain a single ``measure_execution`` method that can wrap
    *any* function or method, making the timing/monitoring infrastructure
    reusable without coupling to a specific method name.
    """

    def measure_execution(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T | None, float, float]:
        """Run *fn* and return ``(result, elapsed_seconds, peak_memory_mb)``.

        A background thread samples RSS every 50 ms to track the peak.
        If *fn* raises, ``result`` is *None* and the exception is stored
        in ``self.last_execution_error`` so the caller can inspect it.
        The monitor thread is always cleaned up via ``finally``.

        Memory is reported in MiB (1 MiB = 1 048 576 bytes).
        """
        process = psutil.Process(os.getpid())
        mem_base = process.memory_info().rss
        peak_memory = [mem_base]
        keep_measuring = True

        def _monitor() -> None:
            while keep_measuring:
                try:
                    current_mem = process.memory_info().rss
                    if current_mem > peak_memory[0]:
                        peak_memory[0] = current_mem
                    time.sleep(0.05)
                except psutil.NoSuchProcess:
                    break

        monitor_thread = threading.Thread(target=_monitor)
        monitor_thread.start()

        tic = time.time()
        result: T | None = None
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self.last_execution_error = exc  # type: ignore[attr-defined]
        finally:
            keep_measuring = False
            monitor_thread.join()

        elapsed = time.time() - tic
        memory_mb = (peak_memory[0] - mem_base) / (1024 * 1024)

        return result, elapsed, memory_mb
