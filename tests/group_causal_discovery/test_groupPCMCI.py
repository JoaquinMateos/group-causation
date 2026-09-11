import numpy as np
import pytest
import torch

from group_causation.group_causal_discovery.groupPCMCI import (
    GroupPCMCICausalDiscovery,
)


@pytest.fixture
def data() -> np.ndarray:
    return np.random.randn(200, 6).astype(np.float64)


@pytest.fixture
def groups() -> list[set[int]]:
    return [{0, 1}, {2, 3}, {4, 5}]


class TestGroupPCMCIInit:
    def test_default_initialization(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupPCMCICausalDiscovery(
            data, groups, tau_max=2, pc_alpha=0.05, max_conds_dim=2, u=None,
        )
        assert inst.tau_max == 2
        assert inst.pc_alpha == 0.05
        assert inst.max_conds_dim == 2

    def test_init_with_time_index_u(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupPCMCICausalDiscovery(
            data, groups, tau_max=2, u="time_index", num_chunks_of_time_index=5,
        )
        assert inst.u is not None
        assert inst.u.shape == (data.shape[0], 5)

    def test_init_raises_on_bad_ci_test(self, data: np.ndarray, groups: list[set[int]]):
        with pytest.raises(ValueError, match="Unsupported independence test"):
            GroupPCMCICausalDiscovery(
                data, groups, tau_max=2, conditional_independence_test="nonexistent",
                u=None,
            )

    def test_init_with_numpy_u(self, data: np.ndarray, groups: list[set[int]]):
        u_np = np.random.randint(0, 2, size=(data.shape[0], 3)).astype(np.float64)
        inst = GroupPCMCICausalDiscovery(data, groups, tau_max=2, u=u_np)
        assert isinstance(inst.u, torch.Tensor)
        assert inst.u.shape == (data.shape[0], 3)

    def test_device_is_cpu_when_no_gpu(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupPCMCICausalDiscovery(data, groups, tau_max=2, u=None)
        assert str(inst.device) == "cpu"


class TestGroupPCMCIExtractParents:
    def test_extract_parents_returns_dict(
        self, data: np.ndarray, groups: list[set[int]]
    ):
        inst = GroupPCMCICausalDiscovery(
            data, groups, tau_max=2, pc_alpha=0.99, max_conds_dim=2, u=None,
        )
        parents = inst.extract_parents()
        assert isinstance(parents, dict)
        for j in range(len(groups)):
            assert j in parents

    def test_extract_parents_with_many_independencies(self, data, groups):
        inst = GroupPCMCICausalDiscovery(data, groups, tau_max=2, pc_alpha=0.99, max_conds_dim=2, u=None)
        parents = inst.extract_parents()
        assert isinstance(parents, dict)


class TestGroupPCMCIEdgeCases:
    def test_not_enough_samples(self):
        data = np.random.randn(10, 3).astype(np.float64)
        groups = [{0}, {1}, {2}]
        inst = GroupPCMCICausalDiscovery(data, groups, tau_max=2, u=None)
        parents = inst.extract_parents()
        for node, p_list in parents.items():
            assert len(p_list) == 0

    def test_many_groups(self):
        data = np.random.randn(300, 20).astype(np.float64)
        groups = [set(range(i, i + 2)) for i in range(0, 20, 2)]
        inst = GroupPCMCICausalDiscovery(
            data, groups, tau_max=1, pc_alpha=0.5, max_conds_dim=1, u=None,
        )
        parents = inst.extract_parents()
        assert len(parents) == 10

    def test_non_stationarity_shift_without_info_raises(
        self, data: np.ndarray, groups: list[set[int]]
    ):
        with pytest.raises(ValueError, match="non_stationarity_info must have type"):
            GroupPCMCICausalDiscovery(
                data, groups, tau_max=2, u="non_stationarity_shift",
            )

    def test_time_index_without_chunks_raises(
        self, data: np.ndarray, groups: list[set[int]]
    ):
        with pytest.raises(ValueError, match="num_chunks_of_time_index must be specified"):
            GroupPCMCICausalDiscovery(
                data, groups, tau_max=2, u="time_index",
            )

    def test_single_group(self):
        data = np.random.randn(100, 3).astype(np.float64)
        groups = [{0, 1, 2}]
        inst = GroupPCMCICausalDiscovery(data, groups, tau_max=2, pc_alpha=0.05, u=None)
        parents = inst.extract_parents()
        assert 0 in parents


class TestGroupPCMCIInternal:
    def test_test_ci_returns_float_tuple(
        self, data: np.ndarray, groups: list[set[int]]
    ):
        inst = GroupPCMCICausalDiscovery(data, groups, tau_max=2, pc_alpha=0.05, u=None)
        stat, pval = inst._test_ci(x_var=0, x_lag=1, y_var=1, y_lag=0, z_list=[(2, 1)])
        assert isinstance(stat, float)
        assert isinstance(pval, float)
        assert 0.0 <= pval <= 1.0

    def test_test_ci_returns_independence_short_samples(self):
        data = np.random.randn(8, 3).astype(np.float64)
        groups = [{0}, {1}, {2}]
        inst = GroupPCMCICausalDiscovery(data, groups, tau_max=2, u=None)
        stat, pval = inst._test_ci(x_var=0, x_lag=1, y_var=1, y_lag=0, z_list=[])
        assert pval == 1.0

    def test_raw_group_data_tensor(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupPCMCICausalDiscovery(data, groups, tau_max=2, u=None)
        for t in inst._raw_group_data:
            assert isinstance(t, torch.Tensor)
            assert t.shape[0] == data.shape[0]


class TestCausalInputCompleteness:
    def test_cic_returns_valid_parents(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupPCMCICausalDiscovery(
            data, groups, tau_max=2, pc_alpha=0.05, u=None,
            enforce_causal_input_completeness=True,
        )
        parents = inst.extract_parents()
        assert isinstance(parents, dict)
        for j in range(len(groups)):
            assert j in parents

    def test_cic_produces_same_parents_as_vanilla(self, data: np.ndarray, groups: list[set[int]]):
        inst_vanilla = GroupPCMCICausalDiscovery(
            data, groups, tau_max=2, pc_alpha=0.05, u=None,
            enforce_causal_input_completeness=False,
        )
        parents_vanilla = inst_vanilla.extract_parents()

        inst_cic = GroupPCMCICausalDiscovery(
            data, groups, tau_max=2, pc_alpha=0.05, u=None,
            enforce_causal_input_completeness=True,
        )
        parents_cic = inst_cic.extract_parents()

        # Parents should be identical — CIC only augments internal CI records
        assert parents_vanilla == parents_cic

    def test_cic_builds_time_indexed_graph(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupPCMCICausalDiscovery(
            data, groups, tau_max=2, pc_alpha=0.05, u=None,
            enforce_causal_input_completeness=True,
        )
        parents = inst.extract_parents()
        adj = inst._build_time_indexed_graph(parents)
        assert len(adj) > 0
        # All lag-0 nodes should be present
        for j in range(len(groups)):
            assert (j, 0) in adj

    def test_cic_computes_descendants(self, data: np.ndarray, groups: list[set[int]]):
        inst = GroupPCMCICausalDiscovery(
            data, groups, tau_max=2, pc_alpha=0.05, u=None,
            enforce_causal_input_completeness=True,
        )
        parents = inst.extract_parents()
        adj = inst._build_time_indexed_graph(parents)
        desc = inst._compute_descendants(adj)
        # No node is a descendant of itself
        for node in adj:
            assert node not in desc[node]
