import numpy as np
import pytest

from group_causation.data_management.create_toy_datasets import (
    CausalDataset,
    _extract_subgraph,
    get_parents_dict,
)


class TestCausalDatasetInit:
    def test_empty_initialization(self):
        ds = CausalDataset()
        assert ds.time_series is None
        assert ds.parents_dict is None
        assert ds.groups is None
        assert ds.non_stationarity_info == {"applied": False}

    def test_initialization_with_data(self, sample_causal_dataset):
        ds = CausalDataset(
            time_series=sample_causal_dataset["time_series"],
            parents_dict=sample_causal_dataset["parents_dict"],
            groups=sample_causal_dataset["groups"],
        )
        assert ds.time_series is not None
        assert ds.parents_dict is not None
        assert ds.groups is not None

    def test_groups_property(self):
        ds = CausalDataset()
        assert ds.groups is None
        ds.groups = [[0, 1], [2, 3]]
        assert ds.groups == [[0, 1], [2, 3]]


class TestGenerateToyData:
    def test_generate_toy_data_returns_tuple(self):
        ds = CausalDataset()
        ts, parents = ds.generate_toy_data(
            name="test",
            T=200,
            N_vars=6,
            crosslinks_density=0.5,
            min_lag=1,
            max_lag=2,
            dependency_funcs=["linear"],
            dependency_coeffs=[-0.3, 0.3],
            auto_coeffs=[0.5],
            noise_dists=["gaussian"],
            noise_sigmas=[0.2],
        )
        assert ts.shape == (200, 6)
        assert isinstance(parents, dict)

    def test_generate_toy_data_contemp_fraction_requires_min_lag_0(self):
        ds = CausalDataset()
        with pytest.raises(ValueError, match="If min_lag > 0"):
            ds.generate_toy_data(
                name="test", T=100, N_vars=5, min_lag=1, contemp_fraction=0.5
            )

    def test_generate_toy_data_saves_to_folder(self, tmp_path):
        ds = CausalDataset()
        ds.generate_toy_data(
            name="test_save",
            T=100,
            N_vars=5,
            crosslinks_density=0.3,
            min_lag=1,
            max_lag=2,
            datasets_folder=str(tmp_path),
            dependency_funcs=["linear"],
            dependency_coeffs=[-0.3, 0.3],
            auto_coeffs=[0.5],
            noise_dists=["gaussian"],
            noise_sigmas=[0.2],
        )
        assert (tmp_path / "test_save_data.csv").exists()
        assert (tmp_path / "test_save_parents.txt").exists()

    def test_generate_toy_data_retries_on_nan(self):
        ds = CausalDataset()
        ts, parents = ds.generate_toy_data(
            name="test",
            T=100,
            N_vars=5,
            crosslinks_density=0.3,
            min_lag=1,
            max_lag=2,
            dependency_funcs=["linear"],
            dependency_coeffs=[-0.3, 0.3],
            auto_coeffs=[0.5],
            noise_dists=["gaussian"],
            noise_sigmas=[0.2],
        )
        assert np.all(np.isfinite(ts))


class TestGenerateGroupToyData:
    def test_generate_group_toy_data(self):
        ds = CausalDataset()
        result = ds.generate_group_toy_data(
            name="test_group",
            T=200,
            N_vars=12,
            N_groups=3,
            inner_group_crosslinks_density=0.3,
            outer_group_crosslinks_density=0.4,
            n_node_links_per_group_link=2,
            max_lag=2,
            min_lag=1,
            dependency_funcs=["linear"],
            dependency_coeffs=[-0.3, 0.3],
            auto_coeffs=[0.5],
            noise_dists=["gaussian"],
            noise_sigmas=[0.2],
        )
        ts, parents, groups, node_parents, ns_info = result
        assert ts.shape[1] == 12
        assert len(groups) == 3
        assert isinstance(parents, dict)
        assert isinstance(ns_info, dict)

    def test_generate_group_toy_data_with_contemp(self):
        ds = CausalDataset()
        result = ds.generate_group_toy_data(
            name="test_contemp",
            T=200,
            N_vars=12,
            N_groups=3,
            max_lag=2,
            min_lag=0,
            contemp_fraction=0.3,
        )
        ts, parents, groups, node_parents, ns_info = result
        assert ts.shape[1] == 12
        assert len(groups) == 3

    def test_generate_group_toy_data_saves_to_folder(self, tmp_path):
        ds = CausalDataset()
        ds.generate_group_toy_data(
            name="test_save_groups",
            T=100,
            N_vars=8,
            N_groups=2,
            max_lag=2,
            min_lag=1,
            datasets_folder=str(tmp_path),
        )
        assert (tmp_path / "test_save_groups_data.csv").exists()
        assert (tmp_path / "test_save_groups_groups.txt").exists()


class TestExtractGroupParents:
    def test_extract_group_parents(self):
        ds = CausalDataset()
        ds.groups = [[0, 1], [2, 3], [4]]
        node_parents = {
            0: [(2, -1)],
            1: [(3, 0)],
            2: [(4, -1)],
            3: [],
            4: [],
        }
        result = ds.extract_group_parents(node_parents)
        assert 0 in result
        assert all(isinstance(v, list) for v in result.values())

    def test_extract_group_parents_removes_self_loops(self):
        ds = CausalDataset()
        ds.groups = [[0, 1], [2]]
        node_parents = {0: [(1, 0)], 1: [], 2: []}
        result = ds.extract_group_parents(node_parents)
        for node, parents in result.items():
            for p, lag in parents:
                assert not (p == node and lag == 0)

    def test_extract_group_parents_no_groups_raises(self):
        ds = CausalDataset()
        with pytest.raises(AssertionError):
            ds.extract_group_parents({0: [(1, -1)]})


class TestExtractSubgraph:
    def test_basic_subgraph(self):
        parents = {0: [(1, -1)], 1: [(2, -1)], 2: []}
        chosen = [0, 2]
        result = _extract_subgraph(parents, chosen)
        assert 0 in result
        assert 1 in result
        assert len(result) == 2

    def test_subgraph_removes_duplicates(self):
        parents = {0: [(1, -1), (2, -1)], 1: [(2, -1)], 2: []}
        chosen = [0, 2]
        result = _extract_subgraph(parents, chosen)
        assert len(result[0]) == 2

    def test_subgraph_preserves_contemp_edge(self):
        parents = {0: [(1, 0), (2, -1)], 1: [], 2: []}
        chosen = [0, 1]
        result = _extract_subgraph(parents, chosen)
        assert (1, 0) in result[0]

    def test_empty_chosen_returns_empty(self):
        parents = {0: [(1, -1)], 1: []}
        result = _extract_subgraph(parents, [])
        assert result == {}


class TestGetParentsDict:
    def test_tigramite_format(self):
        causal_process = {
            0: [(((0, -1),), 0.5, lambda x: x), (((1, -1),), -0.3, lambda x: x)],
            1: [],
        }
        result = get_parents_dict(causal_process)
        assert 0 in result
        assert (0, -1) in result[0]
        assert (1, -1) in result[0]

    def test_multivariate_format(self):
        causal_process = {
            0: [(((0, -1), (1, -1)), 0.5, lambda x, y: x * y)],
            1: [],
        }
        result = get_parents_dict(causal_process)
        assert (0, -1) in result[0]
        assert (1, -1) in result[0]

    def test_empty_process(self):
        assert get_parents_dict({}) == {}
