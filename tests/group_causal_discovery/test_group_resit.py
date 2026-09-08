import numpy as np
import pytest

from group_causation.group_causal_discovery.group_resit import (
    GroupRESITTimeSeriesCausalDiscovery,
)


@pytest.fixture
def data() -> np.ndarray:
    return np.random.randn(200, 6).astype(np.float64)


@pytest.fixture
def groups() -> list[set[int]]:
    return [{0, 1}, {2, 3}, {4, 5}]


class TestGroupRESITInit:
    def test_default_params(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupRESITTimeSeriesCausalDiscovery(data, groups)
        assert inst.max_lag == 1
        assert inst.epochs == 200

    def test_custom_params(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupRESITTimeSeriesCausalDiscovery(
            data, groups, max_lag=3, epochs=100, hidden_dim=64,
        )
        assert inst.max_lag == 3
        assert inst.epochs == 100
        assert inst.hidden_dim == 64


class TestGroupRESITExtractParents:
    @pytest.mark.slow
    def test_basic_extraction(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupRESITTimeSeriesCausalDiscovery(data, groups)
        parents = inst.extract_parents()
        assert isinstance(parents, dict)
        assert all(isinstance(k, int) for k in parents)

    def test_min_lag_more_than_max_raises(self, data: np.ndarray):
        groups_single = [{0}, {1}, {2}, {3}, {4}, {5}]
        with pytest.raises(ValueError, match="min_lag cannot be strictly greater than max_lag"):
            GroupRESITTimeSeriesCausalDiscovery(
                data, groups_single, min_lag=3, max_lag=1,
            )


class TestGroupRESITParentsValid:
    @pytest.mark.slow
    def test_identified_parents_are_valid(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupRESITTimeSeriesCausalDiscovery(data, groups)
        parents = inst.extract_parents()
        for group_idx, parent_list in parents.items():
            assert isinstance(group_idx, int)
            for parent, lag in parent_list:
                assert isinstance(parent, int)
                assert isinstance(lag, int)
