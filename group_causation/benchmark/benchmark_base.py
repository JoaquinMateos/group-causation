from abc import ABC, abstractmethod
import logging
import os
from importlib_metadata import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from typing import Any, Iterator, Mapping, Optional, Union
from matplotlib.axes import Axes
from tqdm import tqdm

# Inner library imports
from group_causation.groups_extraction.causal_groups_extraction import CausalGroupsExtractorBase
from group_causation.groups_extraction.stat_utils import get_average_pc1_explained_variance, get_normalized_mutual_information, get_explainability_score
from group_causation.create_toy_datasets import CausalDataset, plot_ts_graph
from group_causation.utils import get_FN, get_FP, get_TP, get_f1, get_precision, get_recall, get_shd, window_to_summary_graph
from group_causation.micro_causal_discovery.micro_causal_discovery_base import MicroCausalDiscovery
from group_causation.group_causal_discovery.direction_extraction.direction_extraction_base import DirectionExtractorBase
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery

AlgorithmCls = Union[type[MicroCausalDiscovery], type[GroupCausalDiscovery], type[CausalGroupsExtractorBase]]


def _parent_to_node(parent: Union[int, tuple[int, int]]) -> int:
    return parent if isinstance(parent, int) else parent[0]

# For printings
BLUE = '\033[34m'
GREEN = '\033[32m'
RESET = '\033[0m'


class BenchmarkBase(ABC):
    '''Abstract class with functions that are useful to benchmark different kinds of algorithms'''
    def __init__(self):
        self.verbose = 0
        self.results: dict[str, list[dict[str, Any]]] = {}
        self.algorithms: Mapping[str, AlgorithmCls] = {}
        self.all_algorithms_parameters: dict[str, list[dict[str, Any]]] = {}
        
    @abstractmethod
    def test_particular_algorithm_particular_dataset(self, causal_dataset: CausalDataset,
                      causalDiscovery: AlgorithmCls,
                      algorithm_parameters: dict[str, Any],) -> dict[str, Any]:
        '''
        To be implemented by subclasses.
        '''
        pass
    
    @abstractmethod
    def generate_datasets(self, iteration, n_datasets, datasets_folder, data_option):
        '''
        To be implemented by subclasses.
        '''
        pass
    
    @abstractmethod
    def load_datasets(self, datasets_folder):
        '''
        To be implemented by subclasses.
        '''
        pass
    
    def benchmark_causal_discovery(self, 
                                algorithms: Mapping[str, AlgorithmCls],
                                parameters_iterator: Iterator[tuple[dict[str, Any], dict[str, Any]]],
                                datasets_folder: str,
                                results_folder: str,
                                generate_toy_data: bool = False,
                                n_executions: int = 3,
                                verbose: int = 0,
                                )        -> dict[str, list[ dict[str, Any] ]]:
        '''
        Function to execute a series of algorithms over time series datasets,
        using a series of parameters for algorithms and options in the creation of the datasets.
        
        Args:
            algorithms : dict[str, MicroCausalDiscovery]
                A dictionary where keys are the names of the algorithms and values are instances of the algorithms to be tested.
            parameters_iterator : Iterator[tuple[dict[str, Any], dict[str, Any]]]
                An iterator that returns a tuple with the parameters for the algorithms and the options for the datasets.
                Note: If generate_toy_data is False, iterations are used for algorithms parameters.
            generate_toy_data : bool, optional
                If True, the datasets are generated with the options in the parameters_iterator.
                If False, the datasets are taken from the datasets_folder.
            datasets_folder : str, optional
                The name of the folder in which datasets will be saved. If not specified, datasets are not saved.
            results_folder : str
                The name of the folder in which results will be saved. If not specified, results are not saved.
            n_executions : int, optional
                The number of executions for each combination of parameters.
            verbose : int, optional
                The level of comments that is going to be printed.
            
        Returns:
            results: dict[str, list[ dict[str, Any] ]]
                A dictionary where keys are the names of the algorithms and values are 
                    lists with dictionaries containing the results of the benchmark for each algorithm.
        '''
        self.verbose = verbose
        self.results_folder = results_folder
        self.algorithms = algorithms
        self.all_algorithms_parameters = {name: list() for name in algorithms.keys()}
        
        # A list whose items are the lists of dictionaries of results and parameters of the different executions
        self.results = {alg: list() for alg in algorithms.keys()}
        
        if generate_toy_data:
            self.benchmark_with_toy_data(algorithms, parameters_iterator, n_executions, datasets_folder)
        else:
            self.benchmark_with_given_data(algorithms, parameters_iterator, n_executions, datasets_folder)

        return self.results
    
    
    def save_results(self):
        if os.path.exists(self.results_folder):
            # If the folder does not exist, create it
            if not os.path.exists(self.results_folder):
                os.makedirs(self.results_folder)
            # Save the results in a csv file
            for name in self.algorithms.keys():
                # Create a dataframe with the results and the parameters
                df = pd.concat([pd.DataFrame(self.results[name]),
                                pd.DataFrame(self.all_algorithms_parameters[name]) ],
                                axis=1)
                # Set dataset_iteration as the first column
                df.insert(0, 'dataset_iteration', df.pop('dataset_iteration'))
                               
                df.to_csv(f'{self.results_folder}/results_{name}.csv', index=False)
        else:
            # If it does exist, delete previous results
            for filename in os.listdir(self.results_folder):
                if filename.endswith('.csv'):
                    os.remove(f'{self.results_folder}/{filename}')
    
    def benchmark_with_toy_data(self, algorithms: Mapping[str, AlgorithmCls],
                                         parameters_iterator: Iterator[tuple[dict[str, Any], dict[str, Any]]],
                                         n_executions: int,
                                         datasets_folder: str,
                                        )  -> dict[str, list[ dict[str, Any] ]]:
        # Delete previous toy data
        if os.path.exists(datasets_folder):
            for filename in os.listdir(datasets_folder):
                os.remove(f'{datasets_folder}/{filename}')
        else:
            os.makedirs(datasets_folder)

        for iteration, current_parameters in enumerate(parameters_iterator):
            current_algorithms_parameters, data_option = current_parameters
            causal_datasets = self.generate_datasets(iteration, n_executions, datasets_folder, data_option)
            if causal_datasets is None:
                raise ValueError('generate_datasets returned None.')
            
            if self.verbose > 0:
                logging.info('\n' + '-'*50)
                logging.info(BLUE + 'Executing the datasets with option:' + str(data_option) + RESET)
            
            # Generate and save results of all algorithms with current dataset options
            current_results = self.test_algorithms(causal_datasets, algorithms,
                                                   current_algorithms_parameters)
            
            for name, algorithm_results in current_results.items():
                for particular_result in algorithm_results:
                    particular_result.update(data_option) # Include the parameters in the information for results
                    particular_result['dataset_iteration'] = iteration
                
                    # Include current result in the list of result
                    self.results[name].append(particular_result)
                
                    self.all_algorithms_parameters[name].\
                                append(copy.deepcopy(current_algorithms_parameters[name]))
                
            self.save_results()
            if self.verbose > 0:
                logging.info(f'{iteration+1} combinations executed')
            
        return self.results
    
    def benchmark_with_given_data(self, algorithms: Mapping[str, AlgorithmCls],
                                        parameters_iterator: Iterator[tuple[dict[str, Any], dict[str, Any]]],
                                        n_executions_per_data_param: int,
                                        datasets_folder: str,
                                        )  -> dict[str, list[ dict[str, Any] ]]:
        causal_datasets = self.load_datasets(datasets_folder)
        if causal_datasets is None:
            raise ValueError('load_datasets returned None.')
        # Execute the algorithms with the given datasets
        for current_algorithms_parameters, data_option in parameters_iterator:
            if self.verbose > 0:
                logging.info('\n' + '-'*50)
                logging.info(BLUE + 'Datasets have been loaded.' + RESET)
            
            # Generate and save results of all algorithms with given datasets
            current_results = self.test_algorithms(causal_datasets, algorithms,
                                                    current_algorithms_parameters)
            logging.info(f'{current_results=}')
            for name, algorithm_results in current_results.items():
                iteration = -1
                for particular_result in algorithm_results:
                    particular_result.update(data_option) # Include the parameters in the information for results
                    particular_result['dataset_iteration'] = (iteration:=iteration+1) // n_executions_per_data_param

                    # Include current result in the list of result
                    self.results[name].append(particular_result)
            
                    self.all_algorithms_parameters[name].\
                                append(copy.deepcopy(current_algorithms_parameters[name]))
            
            self.save_results()
            
        return self.results
    
    def test_algorithms(self, causal_datasets: list[CausalDataset],
                            algorithms: Mapping[str, AlgorithmCls],
                            algorithms_parameters: dict[str, dict[str, Any]],
                            ) -> dict[str, list[dict[str, Any]]]:
        '''
        Execute the given algorithms and return the results
        '''
        result = dict() # keys are score names and values are the score values
        for name, algorithm in algorithms.items():
            algorithm_results = self.test_particular_algorithm(algorithm_name=name,
                                    causal_datasets=causal_datasets, causalDiscovery=algorithm,
                                    algorithm_parameters=algorithms_parameters[name])
            result[name] = algorithm_results

        return result
    
    def test_particular_algorithm(self, algorithm_name: str,
                                causal_datasets: list[CausalDataset],
                                causalDiscovery: AlgorithmCls,
                                algorithm_parameters: dict[str, Any]) -> list[dict[str, Any]]:
        '''
        Execute the given algorithm n_executions times and return the average and std of the results
        '''
        if len(causal_datasets) <= 0:
            return []
        
        if self.verbose > 0:
            logging.info(GREEN + 'Executing algorithm' + algorithm_name + RESET)
        # Execute the algorithm n_executions times
        results_per_execution = []
        for causal_dataset in tqdm(  causal_datasets ): # tqdm is used to show a progress bar
            results_per_execution.append( self.test_particular_algorithm_particular_dataset(causal_dataset, causalDiscovery, algorithm_parameters) )
            
        if self.verbose > 1:
            logging.info(f'{results_per_execution=}')
            
        return results_per_execution
        

    
    def plot_ts_datasets(self, folder_name):
        files = os.listdir(folder_name)
        data_files = filter(lambda x: x.endswith('.csv'), files)
        for filename in data_files:
            data_name = filename.split('_')[0]
            with open(f'{folder_name}/{data_name}_parents.txt', 'r') as f:
                parents_dict = eval(f.read())
            
            # Plot the time series dataset
            self._plot_ts_dataset(f'{folder_name}/{filename}', parents_dict)
            plt.savefig(f'{folder_name}/{data_name}_plot.pdf')
            plt.close('all')
            
            # Plot the graph structure
            plot_ts_graph(parents_dict, var_names=range(len(parents_dict)))
            plt.savefig(f'{folder_name}/{data_name}_graph.pdf')
            
            # Plot the summary graph structure
            summary_parents = window_to_summary_graph(parents_dict)
            # Make the graph more beautiful by setting parents as past variables
            for son, parents in summary_parents.items():
                summary_parents[son] = [(p[0] if isinstance(p, tuple) else p, -1) for p in parents]
            plot_ts_graph(summary_parents, var_names=range(len(summary_parents)))
            plt.savefig(f'{folder_name}/{data_name}_summary_graph.pdf')
            plt.clf()
            
        
    def _plot_ts_dataset(self, dataset_name, parents_dict):
        '''
        Function to plot the time series dataset and the causal graph
        '''
        dataset = pd.read_csv(dataset_name)
        time_series = dataset.values
        n_variables = time_series.shape[1]
        
        # Plot the time series
        fig, axs = plt.subplots(n_variables, 1, figsize=(10, 3*n_variables))
        axes_list = list(axs) if isinstance(axs, np.ndarray) else [axs]
        for i, variable_name in enumerate(dataset.columns):
            ax: Axes = axes_list[i]
            ax.plot(time_series[:, i], color='red')
            ax.set_title(f'Variable {i}')
            ax.set_xlabel('Time')
            ax.set_ylabel(f'$X^{{{i}}}$')
            ax.grid()
            
            ax.set_ylim(bottom=-6, top=6) # TODO: Delete this line
            
            # Include the parents in the title
            parents = parents_dict.get(int(variable_name), [])
            parents_str = ', '.join([f'$X^{{{p[0]}}}_{{t-{p[1]}}}$' for p in parents])
            ax.set_title(f'$X^{{{i}}}_t$ - Parents: {parents_str}')
        
        plt.subplots_adjust(hspace=0.5)
            
    def plot_moving_results(self, results_folder, scores=['shd', 'f1', 'precision', 'recall', 'time', 'memory'],
                            x_axis='max_lag'):
        '''
        Function to plot the results of the benchmark in when a parameter is varied
        '''
        files = os.listdir(results_folder)
        results_files = filter(lambda x: x.startswith('results_') and x.endswith('.csv'), files)
        get_algo_name = lambda filename: filename.split('_')[1].split('.')[0]
        results_dataframes = {get_algo_name(filename): pd.read_csv(f'{results_folder}/{filename}')\
                                for filename in results_files}
        
        for score in scores:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            for algorithm_name, df_results in results_dataframes.items():
                datasets_groups = [df_results[df_results['dataset_iteration']==i] \
                                    for i in df_results['dataset_iteration'].unique()]
                x = [group[x_axis].mean() for group in datasets_groups]
                std = [group[score].std() for group in datasets_groups]
                y = [group[score].mean() for group in datasets_groups]
                ax.errorbar(x, y, yerr=std, label=algorithm_name,
                             fmt='.-', linewidth=1, capsize=3)
                ax.grid(axis='y')
            ax.set_xlabel(x_axis)
            ax.set_ylabel(score)
            ax.legend()
            
            plt.savefig(f'{results_folder}/plot_{score}.pdf')
            fig.clf(); plt.close('all') # Clear the figure and close it

    def plot_particular_result(self, results_folder,
                                     output_folder=None,
                                     scores=['shd', 'f1', 'precision', 'recall', 'time', 'memory'],
                                     dataset_iteration_to_plot=0):
        '''
        Function to plot the result of the benchmark in a particular configuration (in case 
            the csv files has more than one configuration, the first one is shown)
        '''
        if output_folder is None:
            output_folder = results_folder
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        files = os.listdir(results_folder)
        results_files = filter(lambda x: x.startswith('results_') and x.endswith('.csv'), files)
        get_algo_name = lambda filename: filename.split('_')[1].split('.')[0]
        results_dataframes = {get_algo_name(filename): pd.read_csv(f'{results_folder}/{filename}')\
                                for filename in results_files}
        
        for score in scores:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            all_results = []
            for algorithm_name, df_results in results_dataframes.items():
                if dataset_iteration_to_plot == -1: # Get the results for the last dataset iteration
                    dataset_iteration_to_plot = df_results['dataset_iteration'].max()
                    
                current_dataset_results = df_results.loc[df_results['dataset_iteration'] == dataset_iteration_to_plot].copy()
                current_dataset_results['algorithm'] = algorithm_name
                all_results.append(current_dataset_results)
            
            all_results_df = pd.concat(all_results)
            sns.violinplot(x='algorithm', y=score, data=all_results_df,
                           hue='algorithm',ax=ax)
            ax.grid(axis='y')
            
            
            if score in ['f1', 'precision', 'recall', 
                    'f1_summary', 'precision_summary', 'recall_summary']:
                ax.set_ylim(0, 1)
            if score in ['shd', 'shd_summary']:
                ax.set_ylim(bottom=0, top=None)
                
            algorithms_names = list(results_dataframes.keys())
            ax.set_xticks(range(len(algorithms_names)), algorithms_names)
            ax.set_ylabel(score)
            
            plt.savefig(f'{output_folder}/comparison_{score}.pdf')
            plt.close('all')






# INNER AUXILIAR FUNCTIONS
def _generate_micro_dataset(iteration, n_datasets, datasets_folder, data_option):
        '''
        Function to generate the datasets for the benchmark
        
        Args:
            iteration : int The iteration of the dataset
            n_datasets : int The number of datasets to be generated
            datasets_folder : str The folder in which the datasets will be saved
            data_option : dict[str, Any] The options to generate the datasets
        
        Returns:
            causal_datasets : list[CausalDataset] The list with the datasets
        '''
        # Generate the datasets, with their graph structure and time series
        causal_datasets = [CausalDataset() for _ in range(n_datasets)]
        for current_dataset_index, causal_dataset in enumerate(causal_datasets):
            dataset_index = iteration * n_datasets + current_dataset_index
            causal_dataset.generate_toy_data(dataset_index, datasets_folder=datasets_folder, **data_option)
        
        return causal_datasets

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
    
def _load_micro_datasets(datasets_folder):
        '''
        Function to load the datasets for the benchmark
        
        Args:
            datasets_folder : str The folder in which the datasets are saved
        '''
        causal_datasets = []
        # Obtain datasets from folders acording to their filename: {number}_data.csv
        if os.path.exists(datasets_folder):
            files = [f for f in os.listdir(datasets_folder) if f.endswith('.csv')]
            for filename in sorted(files, key=lambda x: int(x.split('_')[0])):
                if filename.endswith('.csv'):
                    dataset = pd.read_csv(f'{datasets_folder}/{filename}')
                    parents_filename = f'{datasets_folder}/{filename.split("_")[0]}_parents.txt'
                    with open(parents_filename, 'r') as f:
                        parents_dict = eval(f.read())
                    
                    causal_datasets.append(CausalDataset(time_series=dataset.values,
                                                            parents_dict=parents_dict))
        else:
            raise ValueError(f'The dataset folder {datasets_folder} does not exist')
        
        return causal_datasets

def _load_group_datasets(datasets_folder):
        '''
        Function to load the datasets for the benchmark
        
        Args:
            datasets_folder : str The folder in which the datasets are saved
        '''
        causal_datasets = []
        # Obtain datasets from folders acording to their filename: {number}_data.csv
        if os.path.exists(datasets_folder):
            files = [f for f in os.listdir(datasets_folder) if f.endswith('.csv')]
            for filename in sorted(files, key=lambda x: int(x.split('_')[0])):
                if filename.endswith('.csv'):
                    dataset = pd.read_csv(f'{datasets_folder}/{filename}')
                    parents_filename = f'{datasets_folder}/{filename.split("_")[0]}_parents.txt'
                    with open(parents_filename, 'r') as f:
                        parents_dict = eval(f.read())
                    groups_filename = f'{datasets_folder}/{filename.split("_")[0]}_groups.txt'
                    with open(groups_filename, 'r') as f:
                        groups = eval(f.read())
                    
                    causal_datasets.append(CausalDataset(time_series=dataset.values,
                                                            parents_dict=parents_dict,
                                                            groups=groups))
        else:
            raise ValueError(f'The dataset folder {datasets_folder} does not exist')
        
        return causal_datasets
