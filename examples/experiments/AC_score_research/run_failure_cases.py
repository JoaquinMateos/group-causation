"""Benchmark PCA vs iVAE aggregation across the four aggregation-faithfulness failure cases.

Datasets come from ``CausalDataset.generate_latent_macro_data`` (latent macro SCM),
so the benchmark also reports the Mean Correlation Coefficient (MCC) of the
recovered latents against the ground truth.

Usage:
    uv run python examples/experiments/AC_score_research/run_failure_cases.py            # full study
    uv run python examples/experiments/AC_score_research/run_failure_cases.py --quick    # smoke test
    uv run python examples/experiments/AC_score_research/run_failure_cases.py --cases low_variance_causal
"""

import argparse
import copy
import os
from dataclasses import dataclass
from typing import Any

os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from group_causation.benchmark import BenchmarkGroupCausalDiscovery
from group_causation.data_management.latent_macro_scm import CASE_PRESETS
from group_causation.group_causal_discovery import (
    DimensionReductionGroupCausalDiscovery,
    IVAE_GroupPCMCI_Proposal,
)

SEED = 42
FULL_T_VALUES = [200, 1000, 5000]
SCORES = ['mcc', 'shd', 'f1', 'precision', 'recall']

ALGORITHMS = {
    'PCA+GroupPCMCI': DimensionReductionGroupCausalDiscovery,
    'iVAE+GroupPCMCI': IVAE_GroupPCMCI_Proposal,
}


@dataclass(frozen=True)
class ResearchConfig:
    cases: list[str]
    T_values: list[int]
    n_executions: int
    max_parallel_executions: int
    n_groups: int
    micro_dim: int
    max_lag: int
    results_folder: str
    algorithm_parameters: dict[str, dict[str, Any]]


def build_configuration(quick: bool, cases: list[str] | None) -> ResearchConfig:
    vae_params = {
        'batch_size': 64, 'max_epoch': 1_000, 'seed': SEED, 'n_layers': 2,
        'hidden_dim': 128, 'early_stopping_patience': 20, 'lr': 1e-4,
        'activation': 'silu', 'slope': 0.1, 'anneal': False, 'scheduler_tol': 10,
    }
    pcmci_params = {'tau_max': 3, 'pc_alpha': 0.05, 'max_conds_dim': 3}
    pca_params = {'cond_ind_test': 'parcorr', 'min_lag': 1, 'max_lag': 3, 'pc_alpha': 0.05}

    configuration = ResearchConfig(
        cases=cases or list(CASE_PRESETS),
        T_values=FULL_T_VALUES,
        n_executions=3,
        max_parallel_executions=3,
        n_groups=4,
        micro_dim=10,
        max_lag=3,
        results_folder='ac_score_research',
        algorithm_parameters={
            'PCA+GroupPCMCI': {
                'dimensionality_reduction': 'pca',
                'node_causal_discovery_alg': 'pcmci',
                'node_causal_discovery_params': pca_params,
            },
            'iVAE+GroupPCMCI': {
                'u': 'time_index',
                'num_chunks_of_time_index': 10,
                'apply_adag_optimization': False,
                'conditional_independence_test': 'hsic',
                'pcmci_params': pcmci_params,
                'ivae_params': vae_params,
                'verbose': 1,
            },
        },
    )
    return _as_quick(configuration) if quick else configuration


def _as_quick(configuration: ResearchConfig) -> ResearchConfig:
    """Small, fast configuration that exercises the full pipeline end to end."""
    algorithm_parameters = copy.deepcopy(configuration.algorithm_parameters)
    algorithm_parameters['PCA+GroupPCMCI']['node_causal_discovery_params'].update(max_lag=1)
    algorithm_parameters['iVAE+GroupPCMCI']['num_chunks_of_time_index'] = 2
    algorithm_parameters['iVAE+GroupPCMCI']['pcmci_params'].update(tau_max=1, max_conds_dim=1)
    algorithm_parameters['iVAE+GroupPCMCI']['ivae_params'].update(
        batch_size=32, max_epoch=5, n_layers=1, hidden_dim=16, lr=1e-2, early_stopping_patience=None,
    )
    return ResearchConfig(
        cases=configuration.cases,
        T_values=[100],
        n_executions=1,
        max_parallel_executions=1,
        n_groups=2,
        micro_dim=4,
        max_lag=1,
        results_folder=f'{configuration.results_folder}/quick',
        algorithm_parameters=algorithm_parameters,
    )


def _algorithm_parameters_for_case(case: str, config: ResearchConfig) -> dict[str, dict[str, Any]]:
    case_preset = CASE_PRESETS[case]
    latent_dim = int(case_preset.get('latent_dim_per_group', 1))

    algorithm_parameters = copy.deepcopy(config.algorithm_parameters)
    algorithm_parameters['PCA+GroupPCMCI']['pca_n_components'] = latent_dim
    algorithm_parameters['iVAE+GroupPCMCI']['latent_dims'] = [latent_dim] * config.n_groups

    if case_preset.get('non_stationarity_params'):
        # The benchmark forwards the dataset's non-stationarity metadata, so the
        # iVAE can condition its prior on the true regimes instead of time chunks.
        proposal_parameters = algorithm_parameters['iVAE+GroupPCMCI']
        proposal_parameters['u'] = 'non_stationarity_shift'
        proposal_parameters.pop('num_chunks_of_time_index', None)
    return algorithm_parameters


def iter_combinations(case: str, config: ResearchConfig):
    """Yield (algorithm_parameters, data_options) for one case and every sample size."""
    for T in config.T_values:
        yield _algorithm_parameters_for_case(case, config), {
            'generator': 'latent_macro_scm',
            'case': case,
            'n_groups': config.n_groups,
            'micro_dim': config.micro_dim,
            'T': T,
            'max_lag': config.max_lag,
            'seed': SEED,
        }


def run_case(case: str, config: ResearchConfig) -> None:
    case_folder = f'{config.results_folder}/{case}'
    with BenchmarkGroupCausalDiscovery(
        info_file=f'ac_research_info_{case}.log',
        debug_file=f'ac_research_debug_{case}.log',
    ) as benchmark:
        benchmark.benchmark_causal_discovery(
            algorithms=ALGORITHMS,
            parameters_iterator=iter_combinations(case, config),
            datasets_folder=f'{case_folder}/datasets',
            generate_toy_data=True,
            results_folder=case_folder,
            n_executions=config.n_executions,
            max_parallel_executions=config.max_parallel_executions,
            verbose=2,
        )
        benchmark.plot_moving_results(case_folder, x_axis='T', scores=SCORES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true',
                        help='Run a fast smoke configuration (one small dataset per case).')
    parser.add_argument('--cases', nargs='+', choices=list(CASE_PRESETS), default=None,
                        help='Subset of failure cases to run. Defaults to all four.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    research_config = build_configuration(quick=args.quick, cases=args.cases)
    for case_name in research_config.cases:
        run_case(case_name, research_config)
