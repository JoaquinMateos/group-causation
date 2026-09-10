"""Base class for group-level causal discovery algorithms."""

from abc import abstractmethod

import numpy as np

from group_causation.causal_discovery_base import CausalDiscovery


class GroupCausalDiscovery(CausalDiscovery):
    """Abstract base for causal discovery on *groups* of variables.

    Extends ``CausalDiscovery`` by accepting an explicit grouping of
    variables.  When *groups* is *None* every variable is treated as
    its own group.

    Args:
        data: Array of shape ``(n_samples, n_variables)``.
        groups: Sequence of sets, each set listing the variable indices
            that belong to one group.  *None* means one group per variable.
        standarize: When *True* (default) the data is centred and
            scaled per feature before being stored in ``self._data``.
    """

    def __init__(
        self,
        data: np.ndarray,
        groups: list[set[int]] | None = None,
        standarize: bool = True,
        **kwargs,
    ):
        super().__init__(data, standarize, **kwargs)

        if groups is None:
            self._groups = [[i] for i in range(data.shape[1])]
        else:
            self._groups = [list(group) for group in groups]

        self.extra_args = kwargs

    @abstractmethod
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        """Return the parent dict for every *group* of variables."""
