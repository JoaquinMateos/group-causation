"""Base class for group extraction algorithms."""

import logging
from abc import ABC, abstractmethod

import numpy as np

from group_causation.shared_mixins import MemoryMonitorMixin, StandardizationMixin

logger = logging.getLogger(__name__)


class CausalGroupsExtractorBase(StandardizationMixin, MemoryMonitorMixin, ABC):
    """Abstract base for algorithms that propose variable groupings.

    Subclasses must implement ``extract_groups`` which returns a list
    of sets, each set containing the variable indices of one group.

    Args:
        data: Array of shape ``(n_samples, n_variables)``.
        standarize: When *True* (default) the data is centred and
            scaled per feature before being stored in ``self._data``.
    """

    def __init__(self, data: np.ndarray, standarize: bool = True, **kwargs):
        self.initialize_data(data, standarize)
        self.extra_args = kwargs

    @abstractmethod
    def extract_groups(self) -> list[set[int]]:
        """Return the proposed variable groupings."""

    def extract_groups_time_and_memory(
        self,
    ) -> tuple[list[set[int]], float, float]:
        """Run ``extract_groups`` and return ``(groups, elapsed_s, memory_mb)``.

        If the algorithm raises, the error is logged and
        ``( [], elapsed, -1.0 )`` is returned so that benchmark loops
        can continue without crashing.
        """
        groups, elapsed, memory_mb = self.measure_execution(self.extract_groups)

        if groups is None:
            exc = getattr(self, "last_execution_error", None)
            logger.error("Error executing %s: %s", self.__class__.__name__, exc)
            return [], elapsed, -1.0

        return groups, elapsed, memory_mb
