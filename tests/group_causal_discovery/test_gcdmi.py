import numpy as np
import pytest

from group_causation.group_causal_discovery.gcdmi import gCDMICausalDiscovery

FAST_PARAMS = dict(epochs=10, hidden_dim=32, num_layers=1, batch_size=64)


@pytest.fixture
def data() -> np.ndarray:
    return np.random.randn(100, 5).astype(np.float64)


@pytest.fixture
def groups() -> list[set[int]]:
    return [{0, 1}, {2, 3}, {4}]


class TestGcdmiInit:
    def test_default_params(self, data: np.ndarray, groups: list[set[int]]):
        inst = gCDMICausalDiscovery(data, groups, **FAST_PARAMS)
        assert inst.max_lag == 3
        assert inst.alpha == 0.05
        assert inst.epochs == 10

    def test_custom_params(self, data: np.ndarray, groups: list[set[int]]):
        inst = gCDMICausalDiscovery(
            data, groups, max_lag=2, alpha=0.01, epochs=10,
        )
        assert inst.max_lag == 2
        assert inst.alpha == 0.01
        assert inst.epochs == 10


class TestGcdmiExtractParents:
    @pytest.mark.slow
    def test_basic_extraction(self, data: np.ndarray, groups: list[set[int]]):
        inst = gCDMICausalDiscovery(data, groups, **FAST_PARAMS)
        parents = inst.extract_parents()
        assert isinstance(parents, dict)

    def test_ts_too_short_raises(self):
        data = np.random.randn(5, 3).astype(np.float64)
        with pytest.raises(ValueError, match="Time series length T must be strictly greater than max_lag"):
            gCDMICausalDiscovery(data, [{0}, {1}, {2}], max_lag=5)
