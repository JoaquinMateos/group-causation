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

    def test_pca_n_components_controls_embedding_size(self, data: np.ndarray, groups: list[set[int]]):
        inst = DimensionReductionGroupCausalDiscovery(
            data, groups, dimensionality_reduction="pca", pca_n_components=2,
        )
        assert [latent.shape[1] for latent in inst.get_recovered_latents()] == [2, 2, 2]

    def test_pca_n_components_capped_at_group_size(self, data: np.ndarray):
        groups = [{0, 1}, {2, 3, 4, 5}]
        inst = DimensionReductionGroupCausalDiscovery(
            data, groups, dimensionality_reduction="pca", pca_n_components=10,
        )
        assert [latent.shape[1] for latent in inst.get_recovered_latents()] == [2, 4]


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

    def test_component_parents_are_collapsed_to_groups(self, data: np.ndarray):
        groups = [{0, 1}, {2, 3}, {4, 5}]
        inst = DimensionReductionGroupCausalDiscovery(
            data, groups, dimensionality_reduction="pca", pca_n_components=2,
        )
        component_parents = {
            0: [],
            1: [(0, -1)],
            2: [(1, 0)],
            3: [],
            4: [(4, -1), (3, -2)],
            5: [(5, 0)],
        }
        result = inst._convert_component_to_group_parents(component_parents)
        assert result == {
            0: [(0, -1)],
            1: [(0, 0)],
            2: [(2, -1), (1, -2)],
        }
