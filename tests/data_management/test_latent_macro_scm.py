import numpy as np
import pytest
from sklearn.decomposition import PCA

from group_causation.data_management.latent_macro_scm import (
    CASE_PRESETS,
    LatentMacroDataset,
    _build_regime_one_hot,
    generate_latent_macro_scm,
)


def _first_pc_latent_correlation(dataset: LatentMacroDataset, group: int) -> float:
    columns = dataset.groups[group]
    first_pc = PCA(n_components=1).fit_transform(dataset.time_series[:, columns])[:, 0]
    return float(abs(np.corrcoef(first_pc, dataset.latent_true[:, group])[0, 1]))


def _mean_pc_latent_correlation(dataset: LatentMacroDataset) -> float:
    return float(np.mean([_first_pc_latent_correlation(dataset, group) for group in range(len(dataset.groups))]))


class TestShapesAndGroups:
    def test_outputs_are_consistent(self):
        dataset = generate_latent_macro_scm(n_groups=4, micro_dim=6, T=300, seed=0)
        assert dataset.time_series.shape == (300, 24)
        assert dataset.latent_true.shape == (300, 4)
        assert dataset.latent_dims == [1, 1, 1, 1]
        assert len(dataset.groups) == 4
        assert all(len(group) == 6 for group in dataset.groups)

    def test_multiple_latents_per_group(self):
        dataset = generate_latent_macro_scm(n_groups=3, micro_dim=6, T=300, seed=0, latent_dim_per_group=2)
        assert dataset.latent_true.shape == (300, 6)
        assert dataset.latent_dims == [2, 2, 2]
        assert all(len(group) == 6 for group in dataset.groups)

    def test_latent_dim_cannot_exceed_micro_dim(self):
        with pytest.raises(ValueError, match='cannot exceed the group micro_dim'):
            generate_latent_macro_scm(n_groups=2, micro_dim=3, T=100, seed=0, latent_dim_per_group=4)

    def test_nonlinear_mixing_preset_uses_several_latent_sources(self):
        dataset = generate_latent_macro_scm(n_groups=2, micro_dim=6, T=200, seed=0, case='nonlinear_mixing')
        assert dataset.latent_dims == [3, 3]

    def test_groups_partition_all_columns(self):
        dataset = generate_latent_macro_scm(n_groups=3, micro_dim=[2, 5, 3], T=100, seed=0)
        flat = sorted(column for group in dataset.groups for column in group)
        assert flat == list(range(10))

    def test_micro_dim_length_must_match_n_groups(self):
        with pytest.raises(ValueError, match='Expected 3 micro dimensions'):
            generate_latent_macro_scm(n_groups=3, micro_dim=[2, 5], T=100, seed=0)

    def test_non_positive_micro_dim_raises(self):
        with pytest.raises(ValueError, match='Micro dimensions must be >= 1'):
            generate_latent_macro_scm(n_groups=2, micro_dim=[0, 3], T=100, seed=0)

    def test_group_parents_include_autoregression_and_chain(self):
        dataset = generate_latent_macro_scm(n_groups=4, micro_dim=3, T=100, seed=0, max_lag=2)
        for group in range(4):
            assert (group, -1) in dataset.group_parents[group]
        assert (0, -1) in dataset.group_parents[1]
        assert (0, -2) in dataset.group_parents[2]

    def test_generated_values_are_finite(self):
        dataset = generate_latent_macro_scm(n_groups=3, micro_dim=4, T=200, seed=0)
        assert np.all(np.isfinite(dataset.time_series))
        assert np.all(np.isfinite(dataset.latent_true))


class TestDeterminism:
    def test_same_seed_is_reproducible(self):
        first = generate_latent_macro_scm(n_groups=3, micro_dim=4, T=150, seed=7)
        second = generate_latent_macro_scm(n_groups=3, micro_dim=4, T=150, seed=7)
        np.testing.assert_array_equal(first.time_series, second.time_series)
        np.testing.assert_array_equal(first.latent_true, second.latent_true)
        assert first.group_parents == second.group_parents

    def test_different_seeds_differ(self):
        first = generate_latent_macro_scm(n_groups=3, micro_dim=4, T=150, seed=1)
        second = generate_latent_macro_scm(n_groups=3, micro_dim=4, T=150, seed=2)
        assert not np.allclose(first.latent_true, second.latent_true)


class TestLinearVersusNonLinearMixing:
    def test_linear_mixing_is_recoverable_by_pca(self):
        dataset = generate_latent_macro_scm(
            n_groups=3, micro_dim=8, T=800, seed=0,
            mixing_activation=lambda x: x, observation_noise_scale=0.05,
        )
        assert _mean_pc_latent_correlation(dataset) > 0.99

    def test_unknown_mixing_activation_raises(self):
        with pytest.raises(ValueError, match='Unknown mixing activation'):
            generate_latent_macro_scm(n_groups=2, micro_dim=3, T=100, seed=0, mixing_activation='not-an-activation')


class TestCasePresets:
    @pytest.mark.parametrize('case', list(CASE_PRESETS))
    def test_every_case_runs(self, case):
        dataset = generate_latent_macro_scm(n_groups=3, micro_dim=6, T=400, seed=0, case=case)
        assert dataset.case == case
        assert np.all(np.isfinite(dataset.time_series))

    def test_unknown_case_raises(self):
        with pytest.raises(ValueError, match='Unknown case'):
            generate_latent_macro_scm(n_groups=2, micro_dim=3, T=100, seed=0, case='not-a-case')

    def test_low_variance_causal_case_defeats_pca(self):
        dataset = generate_latent_macro_scm(n_groups=3, micro_dim=8, T=800, seed=0, case='low_variance_causal')
        assert _mean_pc_latent_correlation(dataset) < 0.3

    def test_low_variance_case_signal_is_weaker_than_noise(self):
        dataset = generate_latent_macro_scm(n_groups=2, micro_dim=6, T=500, seed=0, case='low_variance_causal')
        latent_std = float(dataset.latent_true.std())
        observed_std = float(dataset.time_series.std())
        assert latent_std < observed_std

    def test_nonstationary_case_provides_regime_labels(self):
        dataset = generate_latent_macro_scm(n_groups=2, micro_dim=4, T=500, seed=0, case='nonstationary_regimes')
        assert dataset.non_stationarity_info['applied']
        assert dataset.u is not None
        assert dataset.u.shape == (500, dataset.non_stationarity_info['num_shifts'] + 1)
        np.testing.assert_allclose(dataset.u.sum(axis=1), 1.0)

    def test_explicit_arguments_override_preset(self):
        dataset = generate_latent_macro_scm(
            n_groups=2, micro_dim=4, T=200, seed=0,
            case='nonstationary_regimes', non_stationarity_params={},
        )
        assert dataset.u is None
        assert not dataset.non_stationarity_info['applied']

    def test_nongaussian_case_uses_non_gaussian_innovations(self):
        dataset = generate_latent_macro_scm(n_groups=2, micro_dim=4, T=200, seed=0, case='nongaussian_innovations')
        assert dataset.time_series.shape == (200, 8)


class TestRegimeOneHot:
    def test_returns_none_without_non_stationarity(self):
        assert _build_regime_one_hot({'applied': False}, T=10) is None

    def test_builds_one_hot_from_shift_details(self):
        info = {
            'applied': True,
            'type': 'regime_shifts',
            'num_shifts': 1,
            'affected_vars': [0],
            'shift_details': {
                0: [
                    {'regime': 0, 'start': 0, 'end': 3, 'mean_shift': 0.0, 'std_mult': 1.0},
                    {'regime': 1, 'start': 3, 'end': 6, 'mean_shift': 1.0, 'std_mult': 1.0},
                ],
            },
        }
        one_hot = _build_regime_one_hot(info, T=6)
        np.testing.assert_array_equal(one_hot.argmax(axis=1), [0, 0, 0, 1, 1, 1])

    def test_returns_none_when_affected_vars_have_no_details(self):
        info = {'applied': True, 'type': 'regime_shifts', 'affected_vars': [], 'shift_details': {}}
        assert _build_regime_one_hot(info, T=10) is None
