import logging
from importlib_metadata import files
import numpy as np
from typing import Any

# Inner library imports
from group_causation.benchmark.benchmark_base import _generate_micro_dataset, _load_micro_datasets, parent_to_node
from group_causation.data_management.create_toy_datasets import CausalDataset
from group_causation.utils import get_FN, get_FP, get_TP, get_cpdag_and_edge_set, get_f1, get_false_positive_ratio, get_precision, get_recall, get_shd, window_to_summary_graph
from group_causation.micro_causal_discovery.micro_causal_discovery_base import MicroCausalDiscovery
from group_causation.benchmark import BenchmarkBase



class BenchmarkCausalDiscovery(BenchmarkBase):
    def __init__(self):
        super().__init__()
    
    def generate_datasets(self, iteration, n_datasets, datasets_folder, data_option):
        '''
        Function to generate the datasets for the micro benchmark
        
        Args:
            n_datasets : int The number of datasets to be generated
            datasets_folder : str The folder in which the datasets will be saved
            data_option : dict[str, Any] The options to generate the datasets
        
        Returns:
            causal_datasets : list[CausalDataset] The list with the datasets
        '''
        
        if self.verbose > 0:
            logging.info('Generating datasets...')
        
        return _generate_micro_dataset(iteration=iteration, n_datasets=n_datasets,
                                       datasets_folder=datasets_folder, data_option=data_option)
    
    def load_datasets(self, datasets_folder):
        '''
        Function to load the datasets for the benchmark
        
        Args:
            datasets_folder : str The folder in which the datasets are saved
        '''
        return _load_micro_datasets(datasets_folder=datasets_folder)
    
    def test_particular_algorithm_particular_dataset(self, causal_dataset: CausalDataset,
                      causalDiscovery: type[MicroCausalDiscovery],
                      algorithm_parameters: dict[str, Any],) -> dict[str, Any]:
        '''
        Execute the algorithm one single time and calculate the necessary scores.
        
        Args:
            causal_dataset : CausalDataset with the time series and the parents
            causalDiscovery : class of the algorithm to be executed
            algorithm_parameters : dictionary with the parameters for the algorithm
        Returns:
            result : dictionary with the scores of the algorithm
        '''
        
        if causal_dataset.time_series is None:
            raise ValueError('CausalDataset.time_series is required for benchmarking.')
        if causal_dataset.parents_dict is None:
            raise ValueError('CausalDataset.parents_dict is required for benchmarking.')

        algorithm = causalDiscovery(data=causal_dataset.time_series, **algorithm_parameters)
        try:
            predicted_parents, time, memory = algorithm.extract_parents_time_and_memory()
        except KeyboardInterrupt:
            logging.warning('KeyboardInterrupt caught. Continuing with the next iteration.')
            logging.info('Returning nan values for this algorithm')
            predicted_parents = {}
            time = np.nan
            memory = np.nan
        except Exception as e:
            logging.exception(f'Error in algorithm {causalDiscovery.__name__}: {e}')
            logging.error('Returning nan values for this algorithm')
            predicted_parents = {}
            time = np.nan
            memory = np.nan
        
        finally:
            result = {'time': time, 'memory': memory}
            actual_parents = causal_dataset.parents_dict
            actual_parents_summary = window_to_summary_graph(actual_parents)
            predicted_parents_window = {
                son: [(parent_to_node(p), -1) for p in parents]
                for son, parents in predicted_parents.items()
            }
            
            # --- WINDOW GRAPH METRICS ---
            # Compute the CPDAGs exactly ONCE
            gt_edges, gt_cpdag = get_cpdag_and_edge_set(actual_parents)
            pred_edges, pred_cpdag = get_cpdag_and_edge_set(predicted_parents_window)
            
            # Calculate metrics instantly using sets
            n_nodes_window = len(gt_cpdag.nodes)
            result['precision'] = get_precision(gt_edges, pred_edges)
            result['recall'] = get_recall(gt_edges, pred_edges)
            result['f1'] = get_f1(gt_edges, pred_edges)
            result['fpr'] = get_false_positive_ratio(gt_edges, pred_edges, n_nodes_window)
            result['shd'] = get_shd(gt_cpdag, pred_cpdag)
            
            
            # --- SUMMARY GRAPH METRICS ---
            actual_parents_summary = window_to_summary_graph(actual_parents)
            predicted_parents_summary = window_to_summary_graph(predicted_parents_window)
            
            # Compute the summary CPDAGs
            gt_summary_edges, gt_summary_cpdag = get_cpdag_and_edge_set(actual_parents_summary)
            pred_summary_edges, pred_summary_cpdag = get_cpdag_and_edge_set(predicted_parents_summary)
            
            # Calculate summary metrics instantly
            n_nodes_summary = len(gt_summary_cpdag.nodes)
            result['precision_summary'] = get_precision(gt_summary_edges, pred_summary_edges)
            result['recall_summary'] = get_recall(gt_summary_edges, pred_summary_edges)
            result['f1_summary'] = get_f1(gt_summary_edges, pred_summary_edges)
            result['fpr_summary'] = get_false_positive_ratio(gt_summary_edges, pred_summary_edges, n_nodes_summary)
            result['shd_summary'] = get_shd(gt_summary_cpdag, pred_summary_cpdag)
            
            return result
        