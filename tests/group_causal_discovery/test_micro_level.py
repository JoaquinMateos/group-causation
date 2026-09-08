import numpy as np
import pytest

from group_causation.group_causal_discovery.micro_level import (
    MicroLevelGroupCausalDiscovery,
)


@pytest.fixture
def data() -> np.ndarray:
    return np.random.randn(200, 6).astype(np.float64)


@pytest.fixture
def groups() -> list[set[int]]:
    return [{0, 1}, {2, 3}, {4, 5}]


class TestMicroLevelInit:
    def test_default_initialization(self, data: np.ndarray, groups: list[set[int]]):
        inst = MicroLevelGroupCausalDiscovery(data, groups)
        assert inst.node_causal_discovery_alg == "pcmci"

    def test_custom_params(self, data: np.ndarray, groups: list[set[int]]):
        inst = MicroLevelGroupCausalDiscovery(
            data, groups,
            node_causal_discovery_params={"min_lag": 1, "max_lag": 2, "pc_alpha": 0.1},
        )
        assert inst.node_causal_discovery_params["max_lag"] == 2


class TestMicroLevelConvertNodeToGroup:
    def test_basic_conversion(self):
        inst = MicroLevelGroupCausalDiscovery(
            np.random.randn(100, 4).astype(np.float64),
            groups=[{0, 1}, {2, 3}],
        )
        node_parents = {0: [(2, -1)], 1: [(3, 0)], 2: [], 3: []}
        result = inst._convert_node_to_group_parents(node_parents)
        assert 0 in result
        assert isinstance(result[0], list)

    def test_node_parent_tuple_handling(self):
        inst = MicroLevelGroupCausalDiscovery(
            np.random.randn(100, 4).astype(np.float64),
            groups=[{0, 1}, {2, 3}],
        )
        node_parents = {0: [(2, -1)], 2: []}
        result = inst._convert_node_to_group_parents(node_parents)
        assert 0 in result
        for p, lag in result[0]:
            assert isinstance(p, int)
            assert isinstance(lag, int)

    def test_single_node_parent(self):
        inst = MicroLevelGroupCausalDiscovery(
            np.random.randn(100, 2).astype(np.float64),
            groups=[{0}, {1}],
        )
        node_parents = {0: [(1, -1)], 1: []}
        result = inst._convert_node_to_group_parents(node_parents)
        assert result[0] == [(1, -1)]


class TestMicroLevelGetAlgorithm:
    def test_pcmci_selection(self, data: np.ndarray, groups: list[set[int]]):
        inst = MicroLevelGroupCausalDiscovery(data, groups)
        alg = inst._getCausalDiscoveryAlgorithm()
        assert alg is not None

    def test_dynotears_selection(self, data: np.ndarray, groups: list[set[int]]):
        inst = MicroLevelGroupCausalDiscovery(
            data, groups,
            node_causal_discovery_alg="dynotears",
            node_causal_discovery_params={"min_lag": 1, "max_lag": 2},
        )
        alg = inst._getCausalDiscoveryAlgorithm()
        assert alg is not None

    def test_unknown_algorithm_raises(self, data: np.ndarray, groups: list[set[int]]):
        inst = MicroLevelGroupCausalDiscovery(data, groups)
        inst.node_causal_discovery_alg = "nonexistent"
        with pytest.raises(ValueError, match="Invalid node causal discovery algorithm"):
            inst._getCausalDiscoveryAlgorithm()
