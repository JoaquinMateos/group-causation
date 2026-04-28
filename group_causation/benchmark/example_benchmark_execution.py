import logging
# import torch

from matplotlib import pyplot as plt
import matplotlib
import numpy as np

import copy
from group_causation.benchmark import BenchmarkGroupCausalDiscovery
import os

from group_causation.utils import changing_N_groups, changing_N_variables, changing_N_vars_per_group, changing_alg_params, changing_latent_confounding_fraction, changing_non_stationarity_params, changing_preselection_alpha, static_parameters
from group_causation.group_causal_discovery import DimensionReductionGroupCausalDiscovery
from group_causation.group_causal_discovery import MicroLevelGroupCausalDiscovery
from group_causation.group_causal_discovery import HybridGroupCausalDiscovery
from group_causation.group_causal_discovery import GroupRESITTimeSeriesCausalDiscovery
from group_causation.group_causal_discovery import gCDMICausalDiscovery
from group_causation.group_causal_discovery import IVAE_GroupPCMCI_Proposal

MIN_LAG = 1
MAX_LAG = 3

algorithms = {
    # 'Group-Embedding': HybridGroupCausalDiscovery,
    # 'Group-Embedding - with shift knowledge': HybridGroupCausalDiscovery,
    # 'PCA+PCMCI': DimensionReductionGroupCausalDiscovery,
    # 'PCA+PCMCI - with shift knowledge': DimensionReductionGroupCausalDiscovery,
    # 'PCA+DYNOTEARS': DimensionReductionGroupCausalDiscovery,
    # 'Micro-Level': MicroLevelGroupCausalDiscovery,
    # 'Micro-Level - with shift knowledge': MicroLevelGroupCausalDiscovery,
    # 'GroupRESIT': GroupRESITTimeSeriesCausalDiscovery,
    # 'gCDMI': gCDMICausalDiscovery,
    # 'Proposal': IVAE_GroupPCMCI_Proposal,
    'Proposal - with shift knowledge': IVAE_GroupPCMCI_Proposal,
}
algorithms_parameters = {
    'PCA+PCMCI': {'dimensionality_reduction': 'pca', 'node_causal_discovery_alg': 'pcmci',
                'num_generated_regimes_if_no_shift_info': 5,
                'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                    'max_lag': MAX_LAG,
                                                    'cond_ind_test': 'localized_parcorr',
                                                    'pc_alpha': 0.05}},
    
    'PCA+DYNOTEARS': {'dimensionality_reduction': 'pca', 'node_causal_discovery_alg': 'dynotears',
                            'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                             'max_lag': MAX_LAG,
                                                             'lambda_w': 0.05, 'lambda_a': 0.05}},
    
    'Micro-Level': {'node_causal_discovery_alg': 'pcmci',
                    'num_generated_regimes_if_no_shift_info': 5,
                    'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                        'max_lag': MAX_LAG,
                                                        'cond_ind_test': 'localized_parcorr', # 'localized_hsic',
                                                        'pc_alpha': 0.2}},
    
    'Group-Embedding': {'dimensionality_reduction': 'pca', 
                
                'num_generated_regimes_if_no_shift_info': 5,
                'dimensionality_reduction_params': {'explained_variance_threshold': 0.3,
                                                   'groups_division_method': 'group_embedding'},
                'node_causal_discovery_alg': 'pcmci',
                
                'apply_adag_optimization': True,
                'conditional_independence_test_for_adag': 'max_corr',
                'pc_alpha_for_adag': 0.05,
                'target_c_ind': 0.6,
                
                'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                 'max_lag': MAX_LAG,
                                                 'cond_ind_test': 'localized_parcorr', # 'localized_hsic',
                                                 'pc_alpha': 0.05},
                'verbose': 2},

    'GroupRESIT': {
                'min_lag': MIN_LAG,
                'max_lag': MAX_LAG,
                'epochs': 500,
                'hidden_dim': 64,
                # MURGS hyperparameters
                'lambda_reg': 2e-2,
                'pruning_threshold': 1e-1,
        },
    'gCDMI': {
                'min_lag': MIN_LAG,
                'max_lag': MAX_LAG,
                'epochs': 300,
                'hidden_dim': 128,
                'num_layers': 2,
                'batch_size': 32,
                'alpha': 0.5,
                'learning_rate': 0.001,
                'lambda_l1': 1e-4},
    
    'Proposal': {
                'u': 'time_index',
                'num_chunks_of_time_index': 5,
                'apply_adag_optimization': False,
                'target_c_ind': 0.6,
                'fallback_latent_dims_fraction': 0.4, # Only used if apply_adag_optimization=False
                'pcmci_params': {
                    'tau_max': MAX_LAG,
                    'pc_alpha': 0.05,
                    'max_conds_dim': 3,
                },
                'ivae_params': {
                    'batch_size': 256,
                    'max_iter': 5000,
                    'seed': 42,
                    'n_layers': 2,
                    'hidden_dim': 128,
                    'lr': 1e-3,
                    'activation': 'lrelu',
                    'slope': 0.1,
                    'discrete': False,
                    'anneal': True,
                    'scheduler_tol': 5,
                },
                'verbose': 1,
    },
}

algorithms_parameters['Group-Embedding - with shift knowledge'] = copy.deepcopy(algorithms_parameters['Group-Embedding'])
algorithms_parameters['Group-Embedding - with shift knowledge']['node_causal_discovery_params']['cond_ind_test'] = 'shift_based_local_parcorr'

algorithms_parameters['PCA+PCMCI - with shift knowledge'] = copy.deepcopy(algorithms_parameters['PCA+PCMCI'])
algorithms_parameters['PCA+PCMCI - with shift knowledge']['node_causal_discovery_params']['cond_ind_test'] = 'shift_based_local_parcorr'

algorithms_parameters['Micro-Level - with shift knowledge'] = copy.deepcopy(algorithms_parameters['Micro-Level'])
algorithms_parameters['Micro-Level - with shift knowledge']['node_causal_discovery_params']['cond_ind_test'] = 'shift_based_local_parcorr'

algorithms_parameters['Proposal - with shift knowledge'] = copy.deepcopy(algorithms_parameters['Proposal'])
algorithms_parameters['Proposal - with shift knowledge']['u'] = 'non_stationarity_shift'



data_generation_options = {
    'T': 2_000, # Number of time points in the dataset
    'N_vars': 50, # Number of variables in the dataset
    'N_groups': 8, # Number of groups in the dataset
    'inner_group_crosslinks_density': 0.2, # Density of possible links between nodes of the same group that are created
    'outer_group_crosslinks_density': 0.3, # Density of possible links between groups that are created (if the groups are connected at group level)
    # Confounding params
    'latent_confounding_fraction': 0, # Fraction of latent confounders at the group level (these are groups that are generated but then hidden, so they create latent confounding between the visible groups)
    'maximum_of_nodes_confounded': 3, # Maximum number of nodes per group that can be affected by a single latent confounder
    
    'n_node_links_per_group_link': 2, # Number of links between nodes of two groups that are connected at group level
    'contemp_fraction': 0, # Fraction of links that are contemporaneous (lag 0)
    'cross_terms_fraction': 0.1, # Fraction of links that are cross-terms (multivariate interactions from multiple parents, instead of simple univariate functions of each parent)
    
    # Dependency functions
    'max_lag': MAX_LAG,
    'min_lag': MIN_LAG,
    'dependency_funcs': [#  lambda x: x,
                        lambda x: np.sin(x),
                        lambda x: 2 * min(x**2, 100), # La correlación de Pearson no detecta relaciones cuadráticas
                        lambda x: 1 / (1 + np.exp(-x)) # Sigmoidal
                         ], # Options: 'linear', 'negative-exponential', 'sin', 'cos', 'step'
    'multivariate_funcs': [lambda x, y: 2 * np.clip(x, -10, 10) * np.clip(y, -10, 10)], # Función multiplicativa con capping para evitar valores extremadamente grandes
    'dependency_coeffs': [-0.3, 0.3], # Coefficients for the parent dependencies (these are the :math:`\\beta_{ij}` in the equation in the docstring of generate_toy_data)
    'auto_coeffs': [0.4], # Coefficients for the auto-dependencies (lags of the same variable)
    'noise_dists': ['gaussian', 'weibull'], # List of noise distributions for each variable (in {'gaussian'}, or a function that generates noise given the number of samples)
    'noise_sigmas': [0.4], # Noise standard deviations for each variable (if noise_dists is 'gaussian', these are the standard deviations of the Gaussian noise)
    'group_links': None,
    
    # Stationarity options
    'non_stationarity_params': {
        'type': 'regime_shifts',
            'fraction': 0.5,            # Fractions of variables affected of variables
            'num_shifts': 2,            # Divides the timeline into 4 equal segments
            'max_mean_mod': 10.0,       # Shifts the base mean anywhere between -max_mean_mod and +max_mean_mod
            'max_std_mod': 5.0          # Scales the variance
        }
}

benchmark_options = {
    'static_parameters': (static_parameters, {}),
    'changing_N_variables': (changing_N_variables,
                                    {'list_N_variables': [5]}),
    
    'changing_preselection_alpha': (changing_preselection_alpha,
                                    {'list_preselection_alpha': [0.01, 0.05, 0.1, 0.2]}),
    
    'changing_N_groups': (changing_N_groups,
                                    {'list_N_groups': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                     'relation_vars_per_group': 5}),
    
    'increasing_N_vars_per_group': (changing_N_vars_per_group,
                                    {'list_N_vars_per_group': [2, 4, 6, 8, 10, 12, 14, 16]}),
    
    'increasing_latent_confounding': (changing_latent_confounding_fraction,
                                    {'list_latent_confounding_fraction': [0, 0.17, 0.33, 0.5]}),

    'increasing_non_stationarity': (changing_non_stationarity_params,
                                    {'list_non_stationarity_params': \
                                        [{'type': 'regime_shifts', 'fraction': fraction, 'num_shifts': num_shifts, 'max_mean_mod': 10.0, 'max_std_mod': 5.0} \
                                            for num_shifts, fraction in zip([2]*4, np.linspace(0, 1, 4))
                                        ],}),
    
    'changing_alg_params': (changing_alg_params,
                                    {'alg_name': 'subgroups',
                                     'list_modifying_algorithms_params': [
                                        {'dimensionality_reduction_params': {'explained_variance_threshold': variance,
                                                                             'groups_division_method': 'subgroups'}}\
                                            for variance in list(np.linspace(0.05, 0.95, 19))]})
}

chosen_option = 'increasing_non_stationarity'



if __name__ == '__main__':
    plt.style.use('default')
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 13
    
    # Para ver todos los logs
    logging.basicConfig(level=logging.DEBUG)
    
    benchmark = BenchmarkGroupCausalDiscovery()
    results_folder = 'results_CAEPIA'
    datasets_folder = f'{results_folder}/toy_data'
    
    generate_toy_data = False
    execute_benchmark = True
    plot_graphs = True
    n_executions = 5
    
    dataset_iteration_to_plot = -1
    plot_x_axis = 'drift_fraction'

    options_generator, options_kwargs = benchmark_options[chosen_option]
    parameters_iterator = options_generator(data_generation_options,
                                                algorithms_parameters,
                                                **options_kwargs)
    if execute_benchmark:
        results = benchmark.benchmark_causal_discovery(algorithms=algorithms,
                                            parameters_iterator=parameters_iterator,
                                            datasets_folder=datasets_folder,
                                            generate_toy_data=generate_toy_data,
                                            results_folder=results_folder,
                                            n_executions=n_executions,
                                            verbose=2)
    elif generate_toy_data:
        # Delete previous toy data
        if os.path.exists(datasets_folder):
            for filename in os.listdir(datasets_folder):
                os.remove(f'{datasets_folder}/{filename}')
        else:
            os.makedirs(datasets_folder)

        for iteration, current_parameters in enumerate(parameters_iterator):
            current_algorithms_parameters, data_option = current_parameters
            causal_datasets = benchmark.generate_datasets(iteration, n_executions, datasets_folder, data_option)
    
    if plot_graphs:
        # benchmark.plot_ts_datasets(datasets_folder)
        matplotlib.use('Agg')
        benchmark.plot_moving_results(results_folder, x_axis=plot_x_axis,
                                      scores=['shd', 'f1', 'precision', 'recall', 'time', 'memory', 'f1_summary', 'shd_summary'])
        # Save results for whole graph scores
        benchmark.plot_particular_result(results_folder,
                                        dataset_iteration_to_plot=dataset_iteration_to_plot)
        # Save results for summary graph scores
        benchmark.plot_particular_result(results_folder, results_folder + '/summary',
                                        scores=[f'{score}_summary' for score in \
                                                        ['shd', 'f1', 'precision', 'recall']],
                                        dataset_iteration_to_plot=dataset_iteration_to_plot)

        benchmark.plot_ts_datasets(datasets_folder)