from abc import ABC, abstractmethod
import logging
import os
from importlib_metadata import files
import numpy as np
from typing import Any

from group_causation.benchmark import BenchmarkBase
from group_causation.benchmark.benchmark_base import _generate_group_dataset, _load_group_datasets
from group_causation.create_toy_datasets import CausalDataset
from group_causation.groups_extraction.causal_groups_extraction import CausalGroupsExtractorBase

# Inner library imports
from group_causation.groups_extraction.causal_groups_extraction import CausalGroupsExtractorBase
from group_causation.groups_extraction.stat_utils import get_average_pc1_explained_variance, get_normalized_mutual_information, get_explainability_score



class BenchmarkGroupsExtraction(BenchmarkBase):
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
                      causalExtraction: type[CausalGroupsExtractorBase],
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
        if causal_dataset.groups is None:
            raise ValueError('CausalDataset.groups is required for groups extraction benchmarking.')

        algorithm = causalExtraction(data=causal_dataset.time_series, **algorithm_parameters)
        predicted_groups: list[set[int]] = [set(group) for group in causal_dataset.groups]
        actual_groups = [set(group) for group in causal_dataset.groups]
        result: dict[str, Any] = {'time': np.nan, 'memory': np.nan}
        try:
            predicted_groups, time, memory = algorithm.extract_groups_time_and_memory()
            result['time'] = time
            result['memory'] = memory

            result['predicted_groups'] = predicted_groups
            result['actual_groups'] = actual_groups
            
            result['average_explained_variance'] = get_average_pc1_explained_variance(causal_dataset.time_series, predicted_groups)
            result['n_groups'] = len(predicted_groups)
            result['explainability_score'] = get_explainability_score(causal_dataset.time_series, predicted_groups)
            result['NMI'] = get_normalized_mutual_information(predicted_groups, actual_groups)
        
        except Exception as e:
            logging.exception(f'Error in algorithm {causalExtraction.__name__}: {e}')
            logging.error('Returning nan values for this algorithm')
            predicted_groups = [set(range(causal_dataset.time_series.shape[1]))]
            result['predicted_groups'] = predicted_groups
            result['actual_groups'] = actual_groups
            result['average_explained_variance'] = np.nan
            result['n_groups'] = len(predicted_groups)
            result['explainability_score'] = np.nan
            result['NMI'] = np.nan
        finally:
            return result
        