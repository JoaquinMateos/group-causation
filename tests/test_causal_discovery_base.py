import numpy as np
import pytest

from group_causation.causal_discovery_base import CausalDiscovery


class _ConcreteDiscovery(CausalDiscovery):
    def __init__(self, data: np.ndarray, standarize: bool = True, **kwargs):
        super().__init__(data, standarize, **kwargs)

    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        return {}


class TestCausalDiscoveryBase:
    def test_standardize_default(self, small_data: np.ndarray):
        inst = _ConcreteDiscovery(small_data, standarize=True)
        mean = inst._data.mean(axis=0)
        std = inst._data.std(axis=0)
        assert np.allclose(mean, np.zeros_like(mean), atol=1e-10)
        assert np.allclose(std, np.ones_like(std), atol=1e-6)

    def test_standardize_does_not_mutate_caller(self, small_data: np.ndarray):
        original = small_data.copy()
        _ConcreteDiscovery(small_data, standarize=True)
        assert np.allclose(small_data, original), "caller's array was mutated"

    def test_standardize_false_preserves_data(self, small_data: np.ndarray):
        original = small_data.copy()
        inst = _ConcreteDiscovery(small_data, standarize=False)
        assert np.allclose(inst._data, original)

    def test_standardize_with_constant_feature(self, constant_feature_data: np.ndarray):
        inst = _ConcreteDiscovery(constant_feature_data, standarize=True)
        mean = inst._data.mean(axis=0)
        const_idx = inst._data.shape[1] - 1
        assert np.allclose(mean[:-1], np.zeros(mean.shape[0] - 1), atol=1e-10)
        assert np.abs(mean[const_idx]) < 1e-10

    def test_zero_variance_data(self, zero_variance_data: np.ndarray):
        inst = _ConcreteDiscovery(zero_variance_data, standarize=True)
        assert np.allclose(inst._data, np.zeros_like(zero_variance_data))

    def test_extract_parents_time_and_memory_returns_tuple(
        self, small_data: np.ndarray, mock_psutil
    ):
        inst = _ConcreteDiscovery(small_data)
        parents, elapsed, memory_mb = inst.extract_parents_time_and_memory()
        assert isinstance(parents, dict)
        assert elapsed >= 0.0
        assert memory_mb >= 0.0

    def test_extract_parents_time_and_memory_catches_exception(self, mock_psutil):
        class FailingDiscovery(_ConcreteDiscovery):
            def extract_parents(self):
                msg = "simulated failure"
                raise RuntimeError(msg)

        inst = FailingDiscovery(np.random.randn(50, 3))
        parents, elapsed, memory_mb = inst.extract_parents_time_and_memory()
        assert parents == {}
        assert memory_mb == -1.0

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            CausalDiscovery(np.random.randn(10, 3))
