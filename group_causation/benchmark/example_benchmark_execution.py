import logging
# import torch

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch

from group_causation.benchmark import BenchmarkGroupCausalDiscovery
import os

from group_causation.utils import changing_N_groups, changing_N_variables, changing_N_vars_per_group, changing_alg_params, changing_latent_confounding_fraction, changing_preselection_alpha, static_parameters
from group_causation.group_causal_discovery import DimensionReductionGroupCausalDiscovery
from group_causation.group_causal_discovery import MicroLevelGroupCausalDiscovery
from group_causation.group_causal_discovery import HybridGroupCausalDiscovery
from group_causation.group_causal_discovery import GroupRESITTimeSeriesCausalDiscovery
from group_causation.group_causal_discovery import gCDMICausalDiscovery
from group_causation.group_causal_discovery import IVAEProposalCausalDiscovery
from group_causation.group_causal_discovery import IVAE_GroupPCMCI_Proposal

MIN_LAG = 1
MAX_LAG = 3

algorithms = {
    'Group-Embedding': HybridGroupCausalDiscovery,
    'Subgroups': HybridGroupCausalDiscovery,
    'PCA+PCMCI': DimensionReductionGroupCausalDiscovery,
    # 'PCA+DYNOTEARS': DimensionReductionGroupCausalDiscovery,
    'Micro-Level': MicroLevelGroupCausalDiscovery,
    # 'GroupRESIT': GroupRESITTimeSeriesCausalDiscovery,
    # 'gCDMI': gCDMICausalDiscovery,
    # 'iVAE-Proposal': IVAEProposalCausalDiscovery,
    'iVAE-GroupPCMCI-Proposal': IVAE_GroupPCMCI_Proposal,
}
algorithms_parameters = {
    'PCA+PCMCI': {'dimensionality_reduction': 'pca', 'node_causal_discovery_alg': 'pcmci',
                            'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                             'max_lag': MAX_LAG,
                                                             'pc_alpha': 0.05}},
    
    'PCA+DYNOTEARS': {'dimensionality_reduction': 'pca', 'node_causal_discovery_alg': 'dynotears',
                            'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                             'max_lag': MAX_LAG,
                                                             'lambda_w': 0.05, 'lambda_a': 0.05}},
    
    'Micro-Level': {'node_causal_discovery_alg': 'pcmci',
                            'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                             'max_lag': MAX_LAG,
                                                             'pc_alpha': 0.05}},
    
    'Group-Embedding': {'dimensionality_reduction': 'pca', 
               'dimensionality_reduction_params': {'explained_variance_threshold': 0.3,
                                                   'groups_division_method': 'group_embedding'},
                'node_causal_discovery_alg': 'pcmci',
                'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                 'max_lag': MAX_LAG,
                                                 'pc_alpha': 0.05},
                'verbose': 2},
    
    'Subgroups': {'dimensionality_reduction': 'pca', 
               'dimensionality_reduction_params': {'explained_variance_threshold': 0.3,
                                                   'groups_division_method': 'subgroups'},
                'node_causal_discovery_alg': 'pcmci',
                'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                 'max_lag': MAX_LAG,
                                                 'pc_alpha': 0.05},
                'verbose': 0},
    'GroupRESIT': {
                'min_lag': MIN_LAG,
                'max_lag': MAX_LAG,
                'epochs': 200,
                'hidden_dim': 64,
                # MURGS hyperparameters
                'lambda_reg': 1e-2,
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
    'iVAE-Proposal': {
                'min_lag': MIN_LAG,
                'max_lag': MAX_LAG,
                # Auxiliary variable u: time index for each sample t
                # Shape must be (n_samples, n_aux_vars)
                'u': None,
                # Proposal-level settings
                'global_latent_dim': 8,
                'group_latent_dims': 3,
                'node_causal_discovery_alg': 'pcmci',
                'node_causal_discovery_params': {
                    'min_lag': MIN_LAG,
                    'max_lag': MAX_LAG,
                    'pc_alpha': 0.05,
                },
                # iVAE network/training settings used by IVAE_wrapper
                'ivae_params': {
                    'batch_size': 128,
                    'max_iter': 1000,
                    'seed': 42,
                    'n_layers': 3,
                    'hidden_dim': 128,
                    'lr': 1e-3,
                    'device': 'cuda' if torch.cuda.is_available() \
                                        else 'mps' if torch.backends.mps.is_available() \
                                        else 'cpu',
                    'activation': 'lrelu',
                    'slope': 0.1,
                    'discrete': False,
                    'inference_dim': 3,
                    'anneal': True,
                    'scheduler_tol': 5,
                },
                'verbose': 1,
    },
    'iVAE-GroupPCMCI-Proposal': {
                # Use generated non-stationarity metadata as iVAE auxiliary context.
                'u': 'non_stationarity_shift',
                'global_latent_dim': 8,
                'group_latent_dims': 3,
                'pcmci_params': {
                    'tau_max': MAX_LAG,
                    'pc_alpha': 0.05,
                    'max_conds_dim': 3,
                },
                'ivae_params': {
                    'batch_size': 128,
                    'max_iter': 1000,
                    'seed': 42,
                    'n_layers': 3,
                    'hidden_dim': 128,
                    'lr': 1e-3,
                    'device': 'cuda' if torch.cuda.is_available() \
                                        else 'mps' if torch.backends.mps.is_available() \
                                        else 'cpu',
                    'activation': 'lrelu',
                    'slope': 0.1,
                    'discrete': False,
                    'anneal': True,
                    'scheduler_tol': 5,
                },
                'verbose': 1,
    },
}

data_generation_options = {
    'T': 2000, # Number of time points in the dataset
    'N_vars': 20, # Number of variables in the dataset
    'N_groups': 6, # Number of groups in the dataset
    'inner_group_crosslinks_density': 0.2, # Density of possible links between nodes of the same group that are created
    'outer_group_crosslinks_density': 0.3, # Density of possible links between groups that are created (if the groups are connected at group level)
    # Confounding params
    'latent_confounding_fraction': 0, # Fraction of latent confounders at the group level (these are groups that are generated but then hidden, so they create latent confounding between the visible groups)
    'maximum_of_nodes_confounded': 4,
    
    'n_node_links_per_group_link': 4, # Number of links between nodes of two groups that are connected at group level
    'contemp_fraction': 0, # Fraction of links that are contemporaneous (lag 0)
    'cross_terms_fraction': 0.2, # Fraction of links that are cross-terms (multivariate interactions from multiple parents, instead of simple univariate functions of each parent)
    
    # Dependency functions
    'max_lag': MAX_LAG,
    'min_lag': MIN_LAG,
    'dependency_funcs': [# 'negative-exponential', 'sin', 'cos',
                         lambda x: x,
                         lambda x: np.sin(x),
                         lambda x: 2 * min(x**2, 100), # La correlación de Pearson no detecta relaciones cuadráticas
                          lambda x: 1 / (1 + np.exp(-x)) # Sigmoidal
                         ], # Options: 'linear', 'negative-exponential', 'sin', 'cos', 'step'
    'multivariate_funcs': [lambda x, y: 2 * min(x * y, 100)], # Función multiplicativa con capping para evitar valores extremadamente grandes
    'dependency_coeffs': [-0.3, -0.2, 0.2, 0.3], # Coefficients for the parent dependencies (these are the :math:`\\beta_{ij}` in the equation in the docstring of generate_toy_data)
    'auto_coeffs': [0.3], # Coefficients for the auto-dependencies (lags of the same variable)
    'noise_dists': ['gaussian', 'weibull'], # List of noise distributions for each variable (in {'gaussian'}, or a function that generates noise given the number of samples)
    'noise_sigmas': [0.2], # Noise standard deviations for each variable (if noise_dists is 'gaussian', these are the standard deviations of the Gaussian noise)
    'group_links': None,
    
    # Stationarity options
    'non_stationarity_params': {}
        # 'type': 'regime_shifts',
        #     'fraction': 0.2,            # Fractions of variables affected of variables
        #     'num_shifts': 3,            # Divides the timeline into 4 equal segments
        #     'max_mean_mod': 10.0,       # Shifts the mean anywhere between -10 and +10
        #     'max_std_mod': 2.0          # Scales the variance anywhere between 0.5x and 2.0x
        # }
    
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
    
    'changing_alg_params': (changing_alg_params,
                                    {'alg_name': 'subgroups',
                                     'list_modifying_algorithms_params': [
                                        {'dimensionality_reduction_params': {'explained_variance_threshold': variance,
                                                                             'groups_division_method': 'subgroups'}}\
                                            for variance in list(np.linspace(0.05, 0.95, 19))]})
}

chosen_option = 'static_parameters'



if __name__ == '__main__':
    plt.style.use('default')
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 13
    
    # Para ver todos los logs
    logging.basicConfig(level=logging.INFO)
    
    benchmark = BenchmarkGroupCausalDiscovery()
    results_folder = 'results'
    datasets_folder = f'{results_folder}/toy_data'
    
    execute_benchmark = True
    generate_toy_data = True
    plot_graphs = True
    n_executions = 5
    
    dataset_iteration_to_plot = 0
    plot_x_axis = 'N_vars'

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
    