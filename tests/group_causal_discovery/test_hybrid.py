import numpy as np
import pytest

from group_causation.group_causal_discovery.hybrid import HybridGroupCausalDiscovery


@pytest.fixture
def data() -> np.ndarray:
    return np.random.randn(200, 6).astype(np.float64)


@pytest.fixture
def groups() -> list[set[int]]:
    return [{0, 1}, {2, 3}, {4, 5}]


@pytest.fixture
def dr_params() -> dict:
    return {"explained_variance_threshold": 0.9}


class TestHybridInit:
    def test_default_params(self, data: np.ndarray, groups: list[set[int]], dr_params: dict):
        inst = HybridGroupCausalDiscovery(
            data, groups,
            dimensionality_reduction_params=dr_params,
            node_algorithm="pcmci",
            group_algorithm="mgm",
        )
        assert inst._node_causal_discovery_alg == "pcmci"


class TestHybridExtractParents:
    def test_with_pcmci_mgm(self, data: np.ndarray, groups: list[set[int]]):
        inst = HybridGroupCausalDiscovery(
            data, groups,
            dimensionality_reduction_params={"explained_variance_threshold": 0.9},
            node_algorithm="pcmci",
            group_algorithm="mgm",
            node_causal_discovery_params={
                "min_lag": 1, "max_lag": 2,
                "cond_ind_test": "localized_parcorr", "pc_alpha": 0.99,
            },
        )
        parents = inst.extract_parents()
        assert isinstance(parents, dict)
