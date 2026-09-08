import numpy as np
import pytest

from group_causation.group_causal_discovery.dimension_reduction import (
    DimensionReductionGroupCausalDiscovery,
)


@pytest.fixture
def data() -> np.ndarray:
    return np.random.randn(200, 6).astype(np.float64)


@pytest.fixture
def groups() -> list[set[int]]:
    return [{0, 1}, {2, 3}, {4, 5}]


class TestDimensionReductionInit:
    def test_default_initialization(self, data: np.ndarray, groups: list[set[int]]):
        inst = DimensionReductionGroupCausalDiscovery(
            data, groups, dimensionality_reduction="pca",
        )
        assert inst._groups_data.shape[1] >= 1

    def test_average_reduction(self, data: np.ndarray, groups: list[set[int]]):
        inst = DimensionReductionGroupCausalDiscovery(
            data, groups, dimensionality_reduction="avg",
        )
        assert inst._groups_data.ndim == 2
        assert inst._groups_data.shape[1] == len(groups)

    def test_unknown_reduction_raises(self, data: np.ndarray, groups: list[set[int]]):
        with pytest.raises(ValueError, match="Invalid dimensionality reduction"):
            DimensionReductionGroupCausalDiscovery(
                data, groups, dimensionality_reduction="unknown",
            )

    def test_with_node_causal_discovery_params(
        self, data: np.ndarray, groups: list[set[int]]
    ):
        inst = DimensionReductionGroupCausalDiscovery(
            data, groups, dimensionality_reduction="pca",
            node_causal_discovery_params={"min_lag": 1, "max_lag": 2},
        )
        assert inst.node_causal_discovery_params["max_lag"] == 2


class TestDimensionReductionExtractParents:
    def test_extract_parents_with_pcmci(
        self, data: np.ndarray, groups: list[set[int]]
    ):
        inst = DimensionReductionGroupCausalDiscovery(
            data, groups, dimensionality_reduction="pca",
            node_causal_discovery_alg="pcmci",
            node_causal_discovery_params={
                "min_lag": 1, "max_lag": 2,
                "cond_ind_test": "localized_parcorr", "pc_alpha": 0.99,
            },
        )
        parents = inst.extract_parents()
        assert isinstance(parents, dict)

    def test_unknown_algorithm_raises(self, data: np.ndarray, groups: list[set[int]]):
        inst = DimensionReductionGroupCausalDiscovery(
            data, groups, dimensionality_reduction="pca",
        )
        inst.node_causal_discovery_alg = "nonexistent"
        with pytest.raises(ValueError, match="Invalid node causal discovery algorithm"):
            inst.extract_parents()
