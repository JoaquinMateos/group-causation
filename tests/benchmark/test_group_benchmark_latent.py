import logging

import numpy as np
import pytest

from group_causation.benchmark.benchmark_group_causal_discovery import (
    BenchmarkGroupCausalDiscovery,
    _generate_group_dataset,
    _split_true_latents,
)
from group_causation.data_management.create_toy_datasets import CausalDataset
from group_causation.group_causal_discovery import (
    DimensionReductionGroupCausalDiscovery,
    IVAE_GroupPCMCI_Proposal,
)

FAST_IVAE = dict(batch_size=32, max_epoch=2, seed=0, n_layers=1, hidden_dim=8, lr=1e-2, anneal=False)


def _make_latent_dataset(T: int = 60) -> CausalDataset:
    dataset = CausalDataset()
    dataset.generate_latent_macro_data('tiny', n_groups=2, micro_dim=3, T=T, seed=0)
    return dataset


class TestBenchmarkLoggingContext:
    def test_foreign_handlers_are_detached_and_restored(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        root_logger = logging.getLogger()
        foreign_handler = logging.StreamHandler()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level
        root_logger.addHandler(foreign_handler)
        try:
            with BenchmarkGroupCausalDiscovery(info_file=None, debug_file=None):
                assert foreign_handler not in root_logger.handlers
            assert foreign_handler in root_logger.handlers
            assert root_logger.level == original_level
        finally:
            root_logger.removeHandler(foreign_handler)
            assert root_logger.handlers == original_handlers


class TestGeneratorDispatch:
    def test_latent_macro_generator_saves_ground_truth(self, tmp_path):
        datasets = _generate_group_dataset(0, 1, str(tmp_path), {
            'generator': 'latent_macro_scm', 'case': 'low_variance_causal',
            'n_groups': 2, 'micro_dim': 3, 'T': 60, 'seed': 0,
        })
        assert len(datasets) == 1
        assert datasets[0].latent_true is not None
        assert (tmp_path / '0_latent_true.csv').exists()

    def test_default_generator_does_not_produce_latents(self, tmp_path):
        datasets = _generate_group_dataset(0, 1, str(tmp_path), {
            'T': 60, 'N_vars': 6, 'N_groups': 2, 'max_lag': 1, 'min_lag': 1,
        })
        assert datasets[0].latent_true is None

    def test_unknown_generator_raises(self, tmp_path):
        with pytest.raises(ValueError, match='Unknown dataset generator'):
            _generate_group_dataset(0, 1, str(tmp_path), {'generator': 'nope'})


class TestSplitTrueLatents:
    def test_uses_per_group_latent_dims(self):
        dataset = CausalDataset()
        dataset.generate_latent_macro_data(
            'multi', n_groups=2, micro_dim=6, T=50, seed=0, latent_dim_per_group=2,
        )
        latents = _split_true_latents(dataset)
        assert [latent.shape[1] for latent in latents] == [2, 2]

    def test_falls_back_to_one_column_per_group(self):
        dataset = CausalDataset()
        dataset.latent_true = np.random.randn(20, 3)
        assert len(_split_true_latents(dataset)) == 3


class TestLatentMCCDispatch:
    def test_missing_algorithm_returns_none(self):
        benchmark = BenchmarkGroupCausalDiscovery(info_file=None, debug_file=None)
        assert benchmark._compute_latent_mcc(None, _make_latent_dataset()) is None

    def test_missing_ground_truth_returns_none(self):
        benchmark = BenchmarkGroupCausalDiscovery(info_file=None, debug_file=None)
        assert benchmark._compute_latent_mcc(object(), CausalDataset()) is None

    def test_algorithm_without_latent_recovery_returns_none(self):
        benchmark = BenchmarkGroupCausalDiscovery(info_file=None, debug_file=None)
        assert benchmark._compute_latent_mcc(object(), _make_latent_dataset()) is None


@pytest.mark.slow
class TestLatentMCCInBenchmark:
    def _run(self, causal_discovery, algorithm_parameters):
        benchmark = BenchmarkGroupCausalDiscovery(info_file=None, debug_file=None)
        return benchmark.test_particular_algorithm_particular_dataset(
            _make_latent_dataset(), causal_discovery, algorithm_parameters,
        )

    def test_proposal_run_reports_mcc(self):
        result = self._run(IVAE_GroupPCMCI_Proposal, {
            'u': 'time_index', 'num_chunks_of_time_index': 2,
            'pcmci_params': {'tau_max': 1, 'pc_alpha': 0.5, 'max_conds_dim': 1},
            'ivae_params': FAST_IVAE,
        })
        assert result['mcc'] is not None
        assert 0.0 <= result['mcc'] <= 1.0
        assert 'shd' in result

    def test_pca_baseline_run_reports_mcc(self):
        result = self._run(DimensionReductionGroupCausalDiscovery, {
            'dimensionality_reduction': 'pca',
            'node_causal_discovery_alg': 'pcmci',
            'node_causal_discovery_params': {
                'cond_ind_test': 'parcorr', 'min_lag': 1, 'max_lag': 1, 'pc_alpha': 0.5,
            },
        })
        assert result['mcc'] is not None
        assert 0.0 <= result['mcc'] <= 1.0
