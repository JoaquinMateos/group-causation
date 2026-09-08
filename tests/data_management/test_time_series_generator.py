import numpy as np
import pytest

from group_causation.data_management.time_series_generator import (
    generate_group_causal_process_structure,
    generate_data_from_causal_process_structure,
    _get_topological_order,
    _check_linear_stationarity,
    _apply_non_stationarity,
)


class TestTopologicalOrder:
    def test_simple_dag(self):
        links = {
            0: [((), 0.0, lambda x: x)],
            1: [(((0, 0),), 0.5, lambda x: x)],
            2: [(((1, 0),), 0.5, lambda x: x)],
        }
        order = _get_topological_order(links, 3)
        assert order == [0, 1, 2]

    def test_cyclic_raises(self):
        links = {
            0: [(((1, 0),), 0.5, lambda x: x)],
            1: [(((0, 0),), 0.5, lambda x: x)],
        }
        with pytest.raises(ValueError, match="Cyclic contemporaneous links"):
            _get_topological_order(links, 2)


class TestCheckStationarity:
    def test_stationary_ar1(self):
        links = {
            0: [(((0, -1),), 0.5, lambda x: x)],
        }
        assert _check_linear_stationarity(links, 1, 1)

    def test_non_stationary(self):
        links = {
            0: [(((0, -1),), 1.5, lambda x: x)],
        }
        assert not _check_linear_stationarity(links, 1, 1)

    def test_max_lag_zero(self):
        links = {
            0: [((), 0.0, lambda x: x)],
        }
        assert _check_linear_stationarity(links, 1, 0)


class TestGenerateGroupCausalProcessStructure:
    def test_basic_generation(self):
        groups = [[0, 1], [2, 3]]
        group_links = {0: [(1, -1)], 1: [(0, -2)]}
        links, latent = generate_group_causal_process_structure(
            groups=groups,
            group_links=group_links,
            n_node_links_per_group_link=1,
            inner_group_density=0.0,
            max_lag=2,
            dependency_funcs=[lambda x: x],
            multivariate_funcs=[lambda x, y: x * y],
            dependency_coeffs=[0.3],
            auto_coeffs=[0.5],
            enforce_autoregression=False,
            seed=42,
            enforce_stationarity=False,
        )
        assert isinstance(links, dict)
        assert isinstance(latent, set)
        assert len(links) == 4

    def test_with_latent_confounding(self):
        groups = [[0, 1], [2, 3]]
        group_links = {0: [(1, -1)]}
        links, latent = generate_group_causal_process_structure(
            groups=groups,
            group_links=group_links,
            n_node_links_per_group_link=1,
            inner_group_density=0.0,
            latent_confounding_fraction=0.25,
            max_lag=2,
            dependency_funcs=[lambda x: x],
            multivariate_funcs=[lambda x, y: x * y],
            dependency_coeffs=[0.3],
            auto_coeffs=[0.5],
            enforce_autoregression=False,
            seed=42,
            enforce_stationarity=False,
        )
        assert len(latent) > 0

    def test_enforce_autoregression_requires_max_lag(self):
        groups = [[0]]
        group_links = {}
        with pytest.raises(ValueError, match="enforce_autoregression=True requires max_lag > 0"):
            generate_group_causal_process_structure(
                groups=groups,
                group_links=group_links,
                max_lag=0,
                enforce_autoregression=True,
                seed=42,
                enforce_stationarity=False,
            )

    def test_too_many_latent_confounders_raises(self):
        groups = [[0], [1]]
        group_links = {}
        with pytest.raises(ValueError, match="Latent confounding fraction is too high"):
            generate_group_causal_process_structure(
                groups=groups,
                group_links=group_links,
                latent_confounding_fraction=0.9,
                max_lag=1,
                enforce_autoregression=False,
                seed=42,
                enforce_stationarity=False,
            )


class TestGenerateDataFromCausalProcess:
    def test_generates_correct_shape(self):
        links = {
            0: [(((0, -1),), 0.5, lambda x: x)],
        }
        data, nonvalid, ns_info = generate_data_from_causal_process_structure(
            links=links, T=100, noise_dists=["gaussian"], noise_sigmas=[0.2]
        )
        assert data.shape == (100, 1)
        assert not nonvalid

    def test_multiple_variables(self):
        links = {
            0: [(((0, -1),), 0.5, lambda x: x)],
            1: [(((1, -1),), 0.3, lambda x: x), (((0, 0),), 0.2, lambda x: x)],
        }
        data, nonvalid, ns_info = generate_data_from_causal_process_structure(
            links=links, T=200, noise_dists=["gaussian", "uniform"], noise_sigmas=[0.2, 0.5]
        )
        assert data.shape == (200, 2)
        assert not nonvalid

    def test_with_non_stationarity(self):
        links = {
            0: [(((0, -1),), 0.5, lambda x: x)],
        }
        ns_params = {
            "type": "regime_shifts",
            "fraction": 1.0,
            "num_shifts": 2,
            "max_mean_mod": 3.0,
            "max_std_mod": 2.0,
        }
        data, nonvalid, ns_info = generate_data_from_causal_process_structure(
            links=links, T=100, noise_dists=["gaussian"], noise_sigmas=[0.2],
            non_stationarity_params=ns_params,
        )
        assert data.shape == (100, 1)
        assert ns_info["applied"]


class TestApplyNonStationarity:
    def test_regime_shifts(self):
        ts = np.random.randn(100, 5)
        params = {
            "type": "regime_shifts",
            "fraction": 0.5,
            "num_shifts": 2,
            "max_mean_mod": 3.0,
            "max_std_mod": 2.0,
        }
        modified, info = _apply_non_stationarity(ts, params)
        assert info["applied"]
        assert modified.shape == ts.shape
        assert len(info["affected_vars"]) == 2

    def test_random_walk(self):
        ts = np.random.randn(100, 3)
        params = {"type": "random_walk", "fraction": 1.0}
        modified, info = _apply_non_stationarity(ts, params)
        assert info["applied"]

    def test_unknown_type(self):
        ts = np.random.randn(50, 2)
        params = {"type": "unknown_type", "fraction": 0.5}
        modified, info = _apply_non_stationarity(ts, params)
        assert not info["applied"]
        assert np.allclose(modified, ts)
