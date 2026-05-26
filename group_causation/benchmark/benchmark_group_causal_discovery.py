import copy
import logging
import os
from typing import Any

import numpy as np
import optuna
import pandas as pd

from group_causation.benchmark.benchmark_base import parent_to_node
from group_causation.benchmark.benchmark_causal_discovery import BenchmarkCausalDiscovery
from group_causation.data_management.create_toy_datasets import CausalDataset
from group_causation.dimensionality_reduction.iVAE.wrappers import IVAE_wrapper
from group_causation.utils import (
    get_cpdag_and_edge_set,
    get_dag_edge_set,
    get_f1,
    get_false_positive_ratio,
    get_global_window_metrics,
    get_precision,
    get_recall,
    get_shd,
    split_lagged_and_contemporaneous,
    window_to_summary_graph,
)


class BenchmarkGroupCausalDiscovery(BenchmarkCausalDiscovery):
    def _split_train_validation(self, time_series: np.ndarray, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError('validation_fraction must be strictly between 0 and 1.')

        split_idx = int(round(time_series.shape[0] * (1.0 - validation_fraction)))
        split_idx = max(1, min(split_idx, time_series.shape[0] - 1))
        return time_series[:split_idx], time_series[split_idx:]

    def _make_time_index_u(self, length: int, num_chunks: int) -> np.ndarray:
        num_chunks = max(1, int(num_chunks))
        chunk_indices = np.repeat(np.arange(num_chunks), int(np.ceil(length / num_chunks)))[:length]
        u_one_hot = np.zeros((length, num_chunks))
        u_one_hot[np.arange(length), chunk_indices] = 1
        return u_one_hot

    def _deep_update(self, base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_update(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _sample_from_search_space(self, trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
        sampled_parameters: dict[str, Any] = {}

        for parameter_name, specification in search_space.items():
            if 'choices' in specification:
                sampled_value = trial.suggest_categorical(parameter_name, specification['choices'])
            elif specification.get('type') == 'float':
                sampled_value = trial.suggest_float(
                    parameter_name,
                    float(specification['low']),
                    float(specification['high']),
                    log=bool(specification.get('log', False)),
                )
            elif specification.get('type') == 'int':
                sampled_value = trial.suggest_float(
                    parameter_name,
                    float(specification['low']),
                    float(specification['high']),
                    log=bool(specification.get('log', False)),
                )
                sampled_value = int(round(sampled_value))
            else:
                raise ValueError(f"Unsupported search-space specification for '{parameter_name}': {specification}")

            if specification.get('cast') == 'int':
                sampled_value = int(round(float(sampled_value)))
                if 'low' in specification:
                    sampled_value = max(int(round(float(specification['low']))), sampled_value)
                if 'high' in specification:
                    sampled_value = min(int(round(float(specification['high']))), sampled_value)

            sampled_parameters[parameter_name] = sampled_value

        return sampled_parameters

    def _score_proposal_candidate(
        self,
        train_data: np.ndarray,
        validation_data: np.ndarray,
        groups_as_sets: list[set[int]],
        candidate_parameters: dict[str, Any],
    ) -> float:
        fallback_fraction = float(candidate_parameters.get('fallback_latent_dims_fraction', 0.33))
        num_chunks = int(round(candidate_parameters.get('num_chunks_of_time_index', 5)))
        num_chunks = max(1, num_chunks)
        ivae_params = copy.deepcopy(candidate_parameters.get('ivae_params', {}))

        train_u = self._make_time_index_u(train_data.shape[0], num_chunks)
        validation_u = self._make_time_index_u(validation_data.shape[0], num_chunks)

        group_scores = []
        for group in groups_as_sets:
            group_cols = sorted(group)
            if not group_cols:
                continue

            latent_dim = max(1, min(int(np.ceil(fallback_fraction * len(group_cols))), len(group_cols)))
            _, _, _, info = IVAE_wrapper(
                train_data[:, group_cols],
                train_u,
                inference_dim=latent_dim,
                **ivae_params,
            )
            reducer = info['reducer']
            group_scores.append(reducer.score_elbo(validation_data[:, group_cols], validation_u))

        if not group_scores:
            return float('-inf')

        return float(np.mean(group_scores))

    def _score_gcdmi_candidate(
        self,
        train_data: np.ndarray,
        validation_data: np.ndarray,
        groups_as_sets: list[set[int]],
        causalDiscovery: Any,
        candidate_parameters: dict[str, Any],
    ) -> float:
        candidate = causalDiscovery(
            data=train_data,
            groups=groups_as_sets,
            **candidate_parameters,
        )
        candidate._train_structure()
        return candidate.score_validation_nll(validation_data)

    def _score_group_resit_candidate(
        self,
        train_data: np.ndarray,
        validation_data: np.ndarray,
        groups_as_sets: list[set[int]],
        causalDiscovery: Any,
        candidate_parameters: dict[str, Any],
    ) -> float:
        candidate = causalDiscovery(
            data=train_data,
            groups=groups_as_sets,
            **candidate_parameters,
        )
        return candidate.score_validation_mse(validation_data)

    def _optimize_hyperparameters(
        self,
        causal_dataset: CausalDataset,
        causalDiscovery: Any,
        algorithm_parameters: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        optimization_spec = copy.deepcopy(algorithm_parameters.get('hyperparameter_optimization', {}))
        if not optimization_spec:
            return copy.deepcopy(algorithm_parameters), {}

        search_space = optimization_spec.get('search_space', {})
        if not search_space:
            return copy.deepcopy(algorithm_parameters), {}

        objective = optimization_spec.get('objective')
        validation_fraction = float(optimization_spec.get('validation_fraction', 0.2))
        n_trials = int(optimization_spec.get('n_trials', 20))
        seed = optimization_spec.get('seed', 42)
        sampler = optuna.samplers.TPESampler(seed=seed)

        if causal_dataset.time_series is None:
            raise ValueError('CausalDataset.time_series is required for hyperparameter optimization.')
        if causal_dataset.groups is None:
            raise ValueError('CausalDataset.groups is required for hyperparameter optimization.')

        train_data, validation_data = self._split_train_validation(causal_dataset.time_series, validation_fraction)
        groups_as_sets = [set(group) for group in causal_dataset.groups]

        base_parameters = copy.deepcopy(algorithm_parameters)
        base_parameters.pop('hyperparameter_optimization', None)

        if objective not in {'elbo', 'nll', 'mse'}:
            raise ValueError(f'Unsupported hyperparameter optimization objective: {objective}')

        direction = 'maximize' if objective == 'elbo' else 'minimize'

        def objective_function(trial: Any) -> float:
            candidate_update = self._sample_from_search_space(trial, search_space)
            trial.set_user_attr('normalized_params', copy.deepcopy(candidate_update))
            candidate_parameters = self._deep_update(base_parameters, candidate_update)

            if objective == 'elbo':
                return self._score_proposal_candidate(train_data, validation_data, groups_as_sets, candidate_parameters)
            if objective == 'nll':
                return self._score_gcdmi_candidate(train_data, validation_data, groups_as_sets, causalDiscovery, candidate_parameters)
            return self._score_group_resit_candidate(train_data, validation_data, groups_as_sets, causalDiscovery, candidate_parameters)

        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective_function, n_trials=n_trials, show_progress_bar=False)

        best_candidate = copy.deepcopy(study.best_trial.user_attrs.get('normalized_params', study.best_trial.params))
        best_parameters = self._deep_update(base_parameters, best_candidate)
        best_score = float(study.best_value)

        optimization_report = {
            'optimization_objective': objective,
            'validation_fraction': validation_fraction,
            'n_trials': n_trials,
            'best_hyperparameters': best_candidate,
            'best_optimization_score': best_score,
        }
        best_parameters['optimization_report'] = optimization_report
        return best_parameters, optimization_report

    def generate_datasets(self, iteration, n_datasets, datasets_folder, data_option):
        '''
        Function to generate the datasets for the benchmark
        '''
        if self.verbose > 0:
            logging.info('Generating datasets...')
        return _generate_group_dataset(iteration, n_datasets, datasets_folder, data_option)

    def load_datasets(self, datasets_folder) -> list[CausalDataset]:
        '''
        Function to load the datasets for the benchmark
        '''
        return _load_group_datasets(datasets_folder)

    def test_particular_algorithm_particular_dataset(
        self,
        causal_dataset: CausalDataset,
        causalDiscovery: Any,
        algorithm_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        '''
        Execute the algorithm one single time and calculate the necessary scores.
        '''
        if causal_dataset.time_series is None:
            raise ValueError('CausalDataset.time_series is required for benchmarking.')
        if causal_dataset.parents_dict is None:
            raise ValueError('CausalDataset.parents_dict is required for benchmarking.')
        if causal_dataset.groups is None:
            raise ValueError('CausalDataset.groups is required for group benchmarking.')

        groups_as_sets = [set(group) for group in causal_dataset.groups]
        current_algorithm_parameters = copy.deepcopy(algorithm_parameters)

        if 'hyperparameter_optimization' in current_algorithm_parameters:
            current_algorithm_parameters, optimization_report = self._optimize_hyperparameters(
                causal_dataset,
                causalDiscovery,
                current_algorithm_parameters,
            )
            logging.info(f"Completed hyperparameter optimization for {causalDiscovery.__name__} with report: {optimization_report}")
        else:
            optimization_report = {}

        predicted_parents = {}
        predicted_parents_window = {}
        predicted_parents_summary = {}
        time = np.nan
        memory = np.nan

        if causal_dataset.non_stationarity_info.get('applied', False):
            non_stationarity_info = copy.deepcopy(causal_dataset.non_stationarity_info)
            current_algorithm_parameters['non_stationarity_info'] = non_stationarity_info

            node_params = current_algorithm_parameters.get('node_causal_discovery_params')
            if isinstance(node_params, dict):
                node_params = copy.deepcopy(node_params)
                node_params['non_stationarity_info'] = non_stationarity_info
                current_algorithm_parameters['node_causal_discovery_params'] = node_params

            if self.verbose > 1:
                logging.info(
                    f"{causalDiscovery.__name__}: propagated non_stationarity_info={non_stationarity_info}"
                )

        try:
            algorithm = causalDiscovery(
                data=causal_dataset.time_series,
                groups=groups_as_sets,
                **current_algorithm_parameters,
            )
            predicted_parents, time, memory = algorithm.extract_parents_time_and_memory()

            predicted_parents_window = {
                son: [
                    (int(p[0]), p[1]) if isinstance(p, tuple) and len(p) == 2
                    else (parent_to_node(p), -1)
                    for p in parents
                ]
                for son, parents in predicted_parents.items()
            }

            predicted_parents_summary = window_to_summary_graph(predicted_parents_window)

            if self.verbose > 1:
                logging.info(
                    f'Algorithm {causalDiscovery.__name__} executed in {time:.3f} seconds and {memory:.3f} MB of memory'
                )

        except KeyboardInterrupt:
            logging.warning(f'Algorithm {causalDiscovery.__name__} interrupted by user')
            raise
        except Exception as e:
            logging.exception(f'Error in algorithm {causalDiscovery.__name__}: {e}')
            logging.error('Returning nan values for this algorithm')
            predicted_parents = {}
            predicted_parents_window = {}
            predicted_parents_summary = {}
            time = np.nan
            memory = np.nan

        result = {'time': time, 'memory': memory}
        if optimization_report:
            result.update(optimization_report)

        actual_parents = {
            son: list(parents)
            for son, parents in causal_dataset.parents_dict.items()
        }

        n_nodes = len(actual_parents)
        actual_parents_summary = window_to_summary_graph(actual_parents)

        if self.verbose > 1:
            logging.info(f'Predicted parents: \t{ {parent: sorted(sons) for parent, sons in predicted_parents.items()} }')
            logging.info(f'Actual parents: \t\t{ {parent: sorted(sons) for parent, sons in actual_parents.items()} }')
            logging.info(f'Predicted parents summary: \t{ {parent: sorted(sons) for parent, sons in predicted_parents_summary.items()} }')
            logging.info(f'Actual parents summary: \t\t{ {parent: sorted(sons) for parent, sons in actual_parents_summary.items()} }')

        gt_lagged, gt_contemp = split_lagged_and_contemporaneous(actual_parents)
        pred_lagged, pred_contemp = split_lagged_and_contemporaneous(predicted_parents_window)

        gt_lagged_edges = get_dag_edge_set(gt_lagged)
        pred_lagged_edges = get_dag_edge_set(pred_lagged)

        result['precision_lagged'] = get_precision(gt_lagged_edges, pred_lagged_edges)
        result['recall_lagged'] = get_recall(gt_lagged_edges, pred_lagged_edges)
        result['f1_lagged'] = get_f1(gt_lagged_edges, pred_lagged_edges)
        result['fpr_lagged'] = get_false_positive_ratio(gt_lagged_edges, pred_lagged_edges, n_nodes)

        gt_contemp_edges, gt_contemp_cpdag = get_cpdag_and_edge_set(gt_contemp)
        pred_contemp_edges, pred_contemp_cpdag = get_cpdag_and_edge_set(pred_contemp)

        result['precision_contemp'] = get_precision(gt_contemp_edges, pred_contemp_edges)
        result['recall_contemp'] = get_recall(gt_contemp_edges, pred_contemp_edges)
        result['f1_contemp'] = get_f1(gt_contemp_edges, pred_contemp_edges)
        result['fpr_contemp'] = get_false_positive_ratio(gt_contemp_edges, pred_contemp_edges, n_nodes)
        result['shd_contemp'] = get_shd(gt_contemp_cpdag, pred_contemp_cpdag)

        global_metrics = get_global_window_metrics(
            gt_lagged_edges,
            pred_lagged_edges,
            gt_contemp_edges,
            pred_contemp_edges,
            gt_contemp_cpdag,
            pred_contemp_cpdag,
        )
        result.update(global_metrics)

        gt_summary_edges, gt_summary_cpdag = get_cpdag_and_edge_set(actual_parents_summary)
        pred_summary_edges, pred_summary_cpdag = get_cpdag_and_edge_set(predicted_parents_summary)

        result['precision_summary'] = get_precision(gt_summary_edges, pred_summary_edges)
        result['recall_summary'] = get_recall(gt_summary_edges, pred_summary_edges)
        result['f1_summary'] = get_f1(gt_summary_edges, pred_summary_edges)
        result['fpr_summary'] = get_false_positive_ratio(gt_summary_edges, pred_summary_edges, n_nodes)
        result['shd_summary'] = get_shd(gt_summary_cpdag, pred_summary_cpdag)

        return result



def _generate_group_dataset(iteration, n_datasets, datasets_folder, data_option):
    '''
    Function to generate the datasets for the benchmark

    Args:
        n_datasets : int The number of datasets to be generated
        datasets_folder : str The folder in which the datasets will be saved
        data_option : dict[str, Any] The options to generate the datasets
    '''
    causal_datasets = [CausalDataset() for _ in range(n_datasets)]
    for current_dataset_index, causal_dataset in enumerate(causal_datasets):
        dataset_index = iteration * n_datasets + current_dataset_index
        causal_dataset.generate_group_toy_data(dataset_index, datasets_folder=datasets_folder, **data_option)

    return causal_datasets



def _load_group_datasets(datasets_folder) -> list[CausalDataset]:
    '''
    Function to load the datasets for the benchmark

    Args:
        datasets_folder : str The folder in which the datasets are saved
    '''
    causal_datasets = []
    if os.path.exists(datasets_folder):
        files = [f for f in os.listdir(datasets_folder) if f.endswith('.csv')]
        for filename in sorted(files, key=lambda x: int(x.split('_')[0])):
            dataset = pd.read_csv(f'{datasets_folder}/{filename}')
            dataset_prefix = filename.split('_')[0]
            parents_filename = f'{datasets_folder}/{dataset_prefix}_parents.txt'
            with open(parents_filename, 'r') as f:
                parents_dict = eval(f.read())
            groups_filename = f'{datasets_folder}/{dataset_prefix}_groups.txt'
            with open(groups_filename, 'r') as f:
                groups = eval(f.read())
            non_stationarity_filename = f'{datasets_folder}/{dataset_prefix}_non_stationarity_info.txt'

            causal_dataset = CausalDataset(
                time_series=dataset.values,
                parents_dict=parents_dict,
                groups=groups,
            )
            if os.path.exists(non_stationarity_filename):
                with open(non_stationarity_filename, 'r') as f:
                    causal_dataset.non_stationarity_info = eval(f.read())

            causal_datasets.append(causal_dataset)
    else:
        raise ValueError(f'The dataset folder {datasets_folder} does not exist')

    return causal_datasets
