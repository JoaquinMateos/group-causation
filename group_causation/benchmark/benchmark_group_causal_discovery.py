import logging
from typing import Any
import numpy as np

# Inner library imports
from group_causation.benchmark import BenchmarkCausalDiscovery
from group_causation.benchmark.benchmark_base import _generate_group_dataset, _load_group_datasets, parent_to_node
from group_causation.create_toy_datasets import CausalDataset
from group_causation.utils import (
    get_cpdag_and_edge_set, get_f1, get_false_positive_ratio, get_precision, 
    get_recall, get_shd, window_to_summary_graph, 
    split_lagged_and_contemporaneous, get_dag_edge_set, get_global_window_metrics # <-- Added here
)
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery


class BenchmarkGroupCausalDiscovery(BenchmarkCausalDiscovery):        
    def generate_datasets(self, iteration, n_datasets, datasets_folder, data_option):
        '''
        Function to generate the datasets for the benchmark
        '''
        if self.verbose > 0:
            logging.info('Generating datasets...')
        return _generate_group_dataset(iteration, n_datasets, datasets_folder, data_option)
    
    def load_datasets(self, datasets_folder):
        '''
        Function to load the datasets for the benchmark
        '''
        return _load_group_datasets(datasets_folder)
        
    def test_particular_algorithm_particular_dataset(self, causal_dataset: CausalDataset,
                      causalDiscovery: type[GroupCausalDiscovery],
                      algorithm_parameters: dict[str, Any],) -> dict[str, Any]:
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
        algorithm = causalDiscovery(data=causal_dataset.time_series, groups=groups_as_sets,  **algorithm_parameters)
        
        try:
            predicted_parents, time, memory = algorithm.extract_parents_time_and_memory()
            
            # Dynamically extract the lag if the algorithm provides it as a tuple (node, lag).
            # If no lag is provided, default to -1.
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
                logging.info(f'Algorithm {causalDiscovery.__name__} executed in {time:.3f} seconds and {memory:.3f} MB of memory')
                logging.info(f'Predicted parents: {predicted_parents}')
                logging.info(f'Predicted parents summary: {predicted_parents_summary}')
                
        except Exception as e:
            logging.exception(f'Error in algorithm {causalDiscovery.__name__}: {e}')
            logging.error('Returning nan values for this algorithm')
            predicted_parents = {}
            predicted_parents_window = {}
            predicted_parents_summary = {}
            time = np.nan
            memory = np.nan
            
        finally:
            result = {'time': time, 'memory': memory}
            actual_parents = causal_dataset.parents_dict
            actual_parents_summary = window_to_summary_graph(actual_parents)
            
            if self.verbose > 1:
                logging.info(f'Actual parents: {actual_parents}')
                logging.info(f'Actual parents summary: {actual_parents_summary}')
                
            n_nodes = len(actual_parents)
            
            # --- 1. SPLIT THE WINDOW GRAPHS ---
            gt_lagged, gt_contemp = split_lagged_and_contemporaneous(actual_parents)
            pred_lagged, pred_contemp = split_lagged_and_contemporaneous(predicted_parents_window)
            
            # --- 2. LAGGED METRICS (Direct DAG) ---
            gt_lagged_edges = get_dag_edge_set(gt_lagged)
            pred_lagged_edges = get_dag_edge_set(pred_lagged)
            
            result['precision_lagged'] = get_precision(gt_lagged_edges, pred_lagged_edges)
            result['recall_lagged'] = get_recall(gt_lagged_edges, pred_lagged_edges)
            result['f1_lagged'] = get_f1(gt_lagged_edges, pred_lagged_edges)
            result['fpr_lagged'] = get_false_positive_ratio(gt_lagged_edges, pred_lagged_edges, n_nodes)
            
            # --- 3. CONTEMPORANEOUS METRICS (MEC / CPDAG) ---
            gt_contemp_edges, gt_contemp_cpdag = get_cpdag_and_edge_set(gt_contemp)
            pred_contemp_edges, pred_contemp_cpdag = get_cpdag_and_edge_set(pred_contemp)
            
            result['precision_contemp'] = get_precision(gt_contemp_edges, pred_contemp_edges)
            result['recall_contemp'] = get_recall(gt_contemp_edges, pred_contemp_edges)
            result['f1_contemp'] = get_f1(gt_contemp_edges, pred_contemp_edges)
            result['fpr_contemp'] = get_false_positive_ratio(gt_contemp_edges, pred_contemp_edges, n_nodes)
            result['shd_contemp'] = get_shd(gt_contemp_cpdag, pred_contemp_cpdag)
            
            global_metrics = get_global_window_metrics(
                gt_lagged_edges, pred_lagged_edges,
                gt_contemp_edges, pred_contemp_edges,
                gt_contemp_cpdag, pred_contemp_cpdag
            )
            result.update(global_metrics)
            
            
            # --- 4. SUMMARY GRAPH METRICS (Evaluated as a single CPDAG) ---
            gt_summary_edges, gt_summary_cpdag = get_cpdag_and_edge_set(actual_parents_summary)
            pred_summary_edges, pred_summary_cpdag = get_cpdag_and_edge_set(predicted_parents_summary)
            
            result['precision_summary'] = get_precision(gt_summary_edges, pred_summary_edges)
            result['recall_summary'] = get_recall(gt_summary_edges, pred_summary_edges)
            result['f1_summary'] = get_f1(gt_summary_edges, pred_summary_edges)
            result['fpr_summary'] = get_false_positive_ratio(gt_summary_edges, pred_summary_edges, n_nodes)
            result['shd_summary'] = get_shd(gt_summary_cpdag, pred_summary_cpdag)
            
            return result