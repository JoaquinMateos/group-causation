import logging
from typing import Any
import numpy as np

# Inner library imports
from group_causation.benchmark import BenchmarkCausalDiscovery
from group_causation.benchmark.benchmark_base import _generate_group_dataset, _load_group_datasets, _parent_to_node
from group_causation.create_toy_datasets import CausalDataset
from group_causation.utils import get_FN, get_FP, get_TP, get_f1, get_precision, get_recall, get_shd, window_to_summary_graph
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery




class BenchmarkGroupCausalDiscovery(BenchmarkCausalDiscovery):        
    def generate_datasets(self, iteration, n_datasets, datasets_folder, data_option):
        '''
        Function to generate the datasets for the benchmark
        
        Args:
            n_datasets : int The number of datasets to be generated
            datasets_folder : str The folder in which the datasets will be saved
            data_option : dict[str, Any] The options to generate the datasets
        '''
        if self.verbose > 0:
            logging.info('Generating datasets...')
        return _generate_group_dataset(iteration, n_datasets, datasets_folder, data_option)
    
    def load_datasets(self, datasets_folder):
        '''
        Function to load the datasets for the benchmark
        
        Args:
            datasets_folder : str The folder in which the datasets are saved
        '''
        return _load_group_datasets(datasets_folder)
        
    def test_particular_algorithm_particular_dataset(self, causal_dataset: CausalDataset,
                      causalDiscovery: type[GroupCausalDiscovery],
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
        if causal_dataset.groups is None:
            raise ValueError('CausalDataset.groups is required for group benchmarking.')

        groups_as_sets = [set(group) for group in causal_dataset.groups]
        algorithm = causalDiscovery(data=causal_dataset.time_series, groups=groups_as_sets,  **algorithm_parameters)
        try:
            predicted_parents, time, memory = algorithm.extract_parents_time_and_memory()
            predicted_parents_window = {
                son: [(_parent_to_node(p), -1) for p in parents]
                for son, parents in predicted_parents.items()
            }
            # Obtain the same metrics in the summary graph
            predicted_parents_summary = window_to_summary_graph(predicted_parents_window)
            if self.verbose > 1:
                logging.info(f'Algorithm {causalDiscovery.__name__} executed in {time:.3f} seconds and {memory:.3f} MB of memory')
                logging.info(f'Predicted parents: {predicted_parents}')
                logging.info(f'Predicted parents summary: {predicted_parents_summary}')
        except Exception as e:
            logging.exception(f'Error in algorithm {causalDiscovery.__name__}: {e}')
            logging.error('Returning nan values for this algorithm')
            predicted_parents = {}
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
            result['TP'] = get_TP(actual_parents, predicted_parents)
            result['FP'] = get_FP(actual_parents, predicted_parents)
            result['FN'] = get_FN(actual_parents, predicted_parents)
            result['precision'] = get_precision(actual_parents, predicted_parents)
            result['recall'] = get_recall(actual_parents, predicted_parents)
            result['f1'] = get_f1(actual_parents, predicted_parents)
            result['shd'] = get_shd(actual_parents, predicted_parents)
            
            result['TP_summary'] = get_TP(actual_parents_summary, predicted_parents_summary)
            result['FP_summary'] = get_FP(actual_parents_summary, predicted_parents_summary)
            result['FN_summary'] = get_FN(actual_parents_summary, predicted_parents_summary)
            result['precision_summary'] = get_precision(actual_parents_summary, predicted_parents_summary)
            result['recall_summary'] = get_recall(actual_parents_summary, predicted_parents_summary)
            result['f1_summary'] = get_f1(actual_parents_summary, predicted_parents_summary)
            result['shd_summary'] = get_shd(actual_parents_summary, predicted_parents_summary)
            
            return result