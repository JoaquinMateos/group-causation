import numpy as np
import pytest

from group_causation.group_causal_discovery.group_causal_discovery_base import (
    GroupCausalDiscovery,
)


class _ConcreteGroupDiscovery(GroupCausalDiscovery):
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        return {}


class TestGroupCausalDiscoveryBase:
    def test_default_groups(self, small_data: np.ndarray):
        inst = _ConcreteGroupDiscovery(small_data, groups=None)
        assert len(inst._groups) == small_data.shape[1]
        for g in inst._groups:
            assert len(g) == 1

    def test_custom_groups(self, small_data: np.ndarray, small_groups: list[set[int]]):
        inst = _ConcreteGroupDiscovery(small_data, groups=small_groups)
        assert len(inst._groups) == 3
        assert inst._groups[0] == [0, 1]

    def test_standardize_default(self, small_data: np.ndarray):
        inst = _ConcreteGroupDiscovery(small_data, standarize=True)
        assert np.allclose(inst._data.mean(axis=0), np.zeros(small_data.shape[1]), atol=1e-10)
        assert np.allclose(inst._data.std(axis=0), np.ones(small_data.shape[1]), atol=1e-6)

    def test_standardize_does_not_mutate_caller(self, small_data: np.ndarray):
        original = small_data.copy()
        _ConcreteGroupDiscovery(small_data, standarize=True)
        assert np.allclose(small_data, original), "caller array was mutated"

    def test_cannot_instantiate_abstract(self, small_data: np.ndarray):
        with pytest.raises(TypeError):
            GroupCausalDiscovery(small_data)

    def test_extra_args_stored(self, small_data: np.ndarray):
        inst = _ConcreteGroupDiscovery(small_data, extra_param=42)
        assert inst.extra_args.get("extra_param") == 42
