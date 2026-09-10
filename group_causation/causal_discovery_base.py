"""Base class for causal discovery algorithms."""

import logging
from abc import ABC, abstractmethod

import numpy as np

from group_causation.shared_mixins import MemoryMonitorMixin, StandardizationMixin

logger = logging.getLogger(__name__)


class CausalDiscovery(StandardizationMixin, MemoryMonitorMixin, ABC):
    """Abstract base for all causal discovery algorithms.

    Subclasses must implement ``extract_parents`` which returns a dict
    mapping each variable index to a list of ``(parent_index, lag)`` tuples.

    Args:
        data: Array of shape ``(n_samples, n_variables)``.
        standarize: When *True* (default) the data is centred and
            scaled per feature before being stored in ``self._data``.
    """

    @abstractmethod
    def __init__(self, data: np.ndarray, standarize: bool = True, **kwargs):
        self.initialize_data(data, standarize)

    @abstractmethod
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        """Return the parent dict for every variable."""

    def extract_parents_time_and_memory(
        self,
    ) -> tuple[dict[int, list[tuple[int, int]]], float, float]:
        """Run ``extract_parents`` and return ``(parents, elapsed_s, memory_mb)``.

        If the algorithm raises, the error is logged and
        ``( {}, elapsed, -1.0 )`` is returned so that benchmark loops
        can continue without crashing.
        """
        parents, elapsed, memory_mb = self.measure_execution(self.extract_parents)

        if parents is None:
            exc = getattr(self, "last_execution_error", None)
            logger.error("Error executing %s: %s", self.__class__.__name__, exc)
            return {}, elapsed, -1.0

        return parents, elapsed, memory_mb
