import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch

import copy
import group_causation
from group_causation.benchmark import BenchmarkGroupCausalDiscovery

from group_causation.group_causal_discovery.groupPCMCI import GroupPCMCICausalDiscovery
from group_causation.utils import changing_N_groups, changing_N_variables, changing_N_vars_per_group, changing_alg_params, changing_latent_confounding_fraction, changing_non_stationarity_params, changing_preselection_alpha, static_parameters
from group_causation.group_causal_discovery import DimensionReductionGroupCausalDiscovery
from group_causation.group_causal_discovery import MicroLevelGroupCausalDiscovery
from group_causation.group_causal_discovery import HybridGroupCausalDiscovery
from group_causation.group_causal_discovery import GroupRESITTimeSeriesCausalDiscovery
from group_causation.group_causal_discovery import gCDMICausalDiscovery
from group_causation.group_causal_discovery import IVAE_GroupPCMCI_Proposal

MIN_LAG = 1
MAX_LAG = 3

EXPERIMENT_NAME = 'sensitivity_analysis'

generate_toy_data = True
execute_benchmark = True
plot_graphs = True
n_executions = 5
max_parallel_executions = 5


algorithms = {
    # 'Adag Embedding': HybridGroupCausalDiscovery,
    # 'Adag Embedding - with shift knowledge': HybridGroupCausalDiscovery,
    # 'PCA+PCMCI': DimensionReductionGroupCausalDiscovery,
    # 'PCA+PCMCI - with shift knowledge': DimensionReductionGroupCausalDiscovery,
    # 'Micro Level': MicroLevelGroupCausalDiscovery,
    # 'Micro Level - with shift knowledge': MicroLevelGroupCausalDiscovery,
    # 'GroupRESIT': GroupRESITTimeSeriesCausalDiscovery,
    # 'GroupRESIT - with shift knowledge': GroupRESITTimeSeriesCausalDiscovery,
    # 'gCDMI': gCDMICausalDiscovery,
    # 'gCDMI - with shift knowledge': gCDMICausalDiscovery,
    # 'Group-PCMCI': GroupPCMCICausalDiscovery,
    # 'Group-PCMCI - with shift knowledge': GroupPCMCICausalDiscovery,
    
    # Solo dejamos las que nos interesan para este benchmark de sensibilidad
    'Proposal': IVAE_GroupPCMCI_Proposal,
    # 'Proposal - with shift knowledge': IVAE_GroupPCMCI_Proposal,
}

algorithms_parameters = {
    'PCA+PCMCI': {'dimensionality_reduction': 'pca', 'node_causal_discovery_alg': 'pcmci',
                'num_generated_regimes_if_no_shift_info': 5,
                'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                'max_lag': MAX_LAG,
                                                'cond_ind_test': 'localized_parcorr',
                                                'pc_alpha': 0.05}},
    
    'Micro Level': {'node_causal_discovery_alg': 'pcmci',
                    'num_generated_regimes_if_no_shift_info': 5,
                    'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                        'max_lag': MAX_LAG,
                                                        'cond_ind_test': 'localized_parcorr',
                                                        'pc_alpha': 0.05}},
    
    'Adag Embedding': {'dimensionality_reduction': 'pca', 
                
                'num_generated_regimes_if_no_shift_info': 5,
                'dimensionality_reduction_params': {'explained_variance_threshold': 0.3,
                                                   'groups_division_method': 'group_embedding'},
                'node_causal_discovery_alg': 'pcmci',
                
                'apply_adag_optimization': True,
                'conditional_independence_test_for_adag': 'max_corr',
                'pc_alpha_for_adag': 0.05,
                'target_c_ind': 0.9,
                
                'node_causal_discovery_params': {'min_lag': MIN_LAG,
                                                 'max_lag': MAX_LAG,
                                                 'cond_ind_test': 'localized_parcorr', 
                                                 'pc_alpha': 0.05},
                'verbose': 2},

    'GroupRESIT': {
                'min_lag': MIN_LAG,
                'max_lag': MAX_LAG,
                'epochs': 500,
                'hidden_dim': 64,
                'lambda_reg': 2e-2,
                'pruning_threshold': 1e-1,
        },
    'gCDMI': {
                'min_lag': MIN_LAG,
                'max_lag': MAX_LAG,
                'epochs': 500,
                'hidden_dim': 128,
                'num_layers': 3,
                'batch_size': 32,
                'alpha': 0.5,
                'learning_rate': 0.001,
                'lambda_l1': 1e-4},
    
    'Group-PCMCI': {
                'tau_max': MAX_LAG,
                'pc_alpha': 0.2,
                'cond_ind_test': 'localized_parcorr',
                'max_conds_dim': 3,
                
                'u': 'time_index',
                'num_chunks_of_time_index': 1,
                'verbose': 1
    },
    
    'Proposal': {
                'u': 'time_index',
                'num_chunks_of_time_index': 10, # Este será sobrescrito en el bucle
                'apply_adag_optimization': False,
                'target_c_ind': 0.6,
                'fallback_latent_dims_fraction': 0.5,
                'pcmci_params': {
                    'tau_max': MAX_LAG,
                    'pc_alpha': 0.2,
                    'max_conds_dim': 3,
                },
                'ivae_params': {
                    'batch_size': 64,
                    'max_epoch': 1_000,
                    'seed': 42,
                    'n_layers': 2,
                    'hidden_dim': 128,
                    'early_stopping_patience': 20,
                    'lr': 1e-4,
                    'activation': 'silu', 
                    'slope': 0.1,
                    'anneal': False,
                    'scheduler_tol': 10,
                },
                'verbose': 1,
    },
}

algorithms_parameters['Adag Embedding - with shift knowledge'] = copy.deepcopy(algorithms_parameters['Adag Embedding'])
algorithms_parameters['Adag Embedding - with shift knowledge']['node_causal_discovery_params']['cond_ind_test'] = 'shift_based_local_parcorr'

algorithms_parameters['PCA+PCMCI - with shift knowledge'] = copy.deepcopy(algorithms_parameters['PCA+PCMCI'])
algorithms_parameters['PCA+PCMCI - with shift knowledge']['node_causal_discovery_params']['cond_ind_test'] = 'shift_based_local_parcorr'

algorithms_parameters['Micro Level - with shift knowledge'] = copy.deepcopy(algorithms_parameters['Micro Level'])
algorithms_parameters['Micro Level - with shift knowledge']['node_causal_discovery_params']['cond_ind_test'] = 'shift_based_local_parcorr'

algorithms_parameters['gCDMI - with shift knowledge'] = copy.deepcopy(algorithms_parameters['gCDMI'])
algorithms_parameters['gCDMI - with shift knowledge']['use_nonstationarity_info'] = True

algorithms_parameters['GroupRESIT - with shift knowledge'] = copy.deepcopy(algorithms_parameters['GroupRESIT'])
algorithms_parameters['GroupRESIT - with shift knowledge']['use_nonstationarity_info'] = True

algorithms_parameters['Group-PCMCI - with shift knowledge'] = copy.deepcopy(algorithms_parameters['Group-PCMCI'])
algorithms_parameters['Group-PCMCI - with shift knowledge']['u'] = 'non_stationarity_shift'

algorithms_parameters['Proposal - with shift knowledge'] = copy.deepcopy(algorithms_parameters['Proposal'])
algorithms_parameters['Proposal - with shift knowledge']['u'] = 'non_stationarity_shift'


data_generation_options = {
    'T': 2_000,
    'N_vars': 50,
    'N_groups': 8,
    'inner_group_crosslinks_density': 0.1,
    'outer_group_crosslinks_density': 0.15,
    'latent_confounding_fraction': 0,
    'maximum_of_nodes_confounded': 3,
    
    'n_node_links_per_group_link': 4,
    'contemp_fraction': 0,
    'cross_terms_fraction': 0.05,
    
    'max_lag': MAX_LAG,
    'min_lag': MIN_LAG,
    'dependency_funcs': [lambda x: x,
                        lambda x: np.sin(x),
                        lambda x: 1 / (1 + np.exp(-x)) - 0.5 
                         ],
    'multivariate_funcs': [
                        lambda x, y: np.clip(2 * x*y, -10, 10), 
                           lambda x, y: np.clip(np.maximum(x, y), -10, 10),
                           ],
    'dependency_coeffs': [-0.5, 0.5],
    'auto_coeffs': [0.6],
    'noise_dists': ['gaussian', 'weibull'],
    'noise_sigmas': [0.2],
    'group_links': None,
    
    'non_stationarity_params': {
        'type': 'regime_shifts',
            'fraction': 1,            
            'num_shifts': 3,            
            'max_mean_mod': 10.0,       
            'max_std_mod': 5.0          
        }
}


def changing_num_chunks_of_time_index(data_generation_options, algorithms_parameters, list_num_chunks):
    for num_chunks in list_num_chunks:
        current_algorithms_parameters = copy.deepcopy(algorithms_parameters)
        current_data_generation_options = copy.deepcopy(data_generation_options)

        # Modificamos el parámetro para el 'Proposal'
        if 'Proposal' in current_algorithms_parameters:
            current_algorithms_parameters['Proposal']['num_chunks_of_time_index'] = num_chunks
        
        yield current_algorithms_parameters, current_data_generation_options


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
                                            for num_shifts, fraction in zip([0, 1, 2, 3], [1]*4)
                                        ],}),
    
    'changing_alg_params': (changing_alg_params,
                                    {'alg_name': 'subgroups',
                                     'list_modifying_algorithms_params': [
                                        {'dimensionality_reduction_params': {'explained_variance_threshold': variance,
                                                                             'groups_division_method': 'subgroups'}}\
                                            for variance in list(np.linspace(0.05, 0.95, 19))]}),
    
    # --- AÑADIMOS EL NUEVO BENCHMARK AQUÍ ---
    'changing_num_chunks': (changing_num_chunks_of_time_index, 
                                    {'list_num_chunks': [4, 7, 10, 13, 16]})  # Valores del 5 al 20 inclusive
}

# SELECCIONAMOS LA OPCIÓN DEL NUEVO BENCHMARK
chosen_option = 'changing_num_chunks'


if __name__ == '__main__':
    plt.style.use('default')
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 13
    
    if torch.backends.mps.is_available(): # type: ignore
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    
    with BenchmarkGroupCausalDiscovery(info_file=f'benchmark_info-{EXPERIMENT_NAME}.log',
                                        debug_file=f'benchmark_debug-{EXPERIMENT_NAME}.log',) as benchmark:
        results_folder = './'
        datasets_folder = f'{results_folder}/toy_data'
        
        dataset_iteration_to_plot = -1
        # CAMBIAMOS EL EJE X DE LAS GRÁFICAS A LA VARIABLE QUE ESTAMOS ANALIZANDO
        plot_x_axis = 'num_chunks_of_time_index'

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
                                                max_parallel_executions=max_parallel_executions,
                                                verbose=2)
        elif generate_toy_data:
            if os.path.exists(datasets_folder):
                for filename in os.listdir(datasets_folder):
                    os.remove(f'{datasets_folder}/{filename}')
            else:
                os.makedirs(datasets_folder)

            for iteration, current_parameters in enumerate(parameters_iterator):
                current_algorithms_parameters, data_option = current_parameters
                causal_datasets = benchmark.generate_datasets(iteration, n_executions, datasets_folder, data_option)
        
        if plot_graphs:
            matplotlib.use('Agg')
            benchmark.plot_moving_results(results_folder, x_axis=plot_x_axis,
                                        scores=['shd', 'f1', 'precision', 'recall', 'time', 'memory', 'f1_summary', 'shd_summary'])
            
            benchmark.plot_particular_result(results_folder,
                                            dataset_iteration_to_plot=dataset_iteration_to_plot)
            
            benchmark.plot_particular_result(results_folder, results_folder + '/summary',
                                            scores=[f'{score}_summary' for score in \
                                                        ['shd', 'f1', 'precision', 'recall']],
                                            dataset_iteration_to_plot=dataset_iteration_to_plot)

            benchmark.plot_ts_datasets(datasets_folder)
    
    print('Finished')