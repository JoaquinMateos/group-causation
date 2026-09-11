import numpy as np
import pytest

from group_causation.group_causal_discovery.group_resit import (
    GroupRESITTimeSeriesCausalDiscovery,
)

FAST_PARAMS = dict(epochs=10, hidden_dim=32, batch_size=64)


@pytest.fixture
def data() -> np.ndarray:
    return np.random.randn(100, 6).astype(np.float64)


@pytest.fixture
def groups() -> list[set[int]]:
    return [{0, 1}, {2, 3}, {4, 5}]


class TestGroupRESITInit:
    def test_default_params(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupRESITTimeSeriesCausalDiscovery(data, groups, **FAST_PARAMS)
        assert inst.max_lag == 1
        assert inst.epochs == 10

    def test_custom_params(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupRESITTimeSeriesCausalDiscovery(
            data, groups, max_lag=3, epochs=10, hidden_dim=32,
        )
        assert inst.max_lag == 3
        assert inst.epochs == 10
        assert inst.hidden_dim == 32


class TestGroupRESITExtractParents:
    @pytest.mark.slow
    def test_basic_extraction(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupRESITTimeSeriesCausalDiscovery(data, groups, **FAST_PARAMS)
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
        inst = GroupRESITTimeSeriesCausalDiscovery(data, groups, **FAST_PARAMS)
        parents = inst.extract_parents()
        for group_idx, parent_list in parents.items():
            assert isinstance(group_idx, int)
            for parent, lag in parent_list:
                assert isinstance(parent, int)
                assert isinstance(lag, int)
