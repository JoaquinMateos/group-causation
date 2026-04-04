from typing import Any, Generator, Mapping, Sequence, Union


# Imports
import numpy as np
import pandas as pd


'''
PARAMETERS GENERATIONS
'''
def static_parameters(options: dict[str, Any], algorithms_parameters: dict[str, Any]) -> Generator[tuple[dict[str, Any], dict[str, Any]], None, None]:

    yield algorithms_parameters, options

def changing_N_variables(
    options: dict[str, Any],
    algorithms_parameters: dict[str, Any],
    list_N_variables: Union[list[int], None] = None
) -> Generator[tuple[dict[str, Any], dict[str, Any]], None, None]:
    if list_N_variables is None:
        list_N_variables = [10, 20, 30, 40, 50]
        
    for N_variables in list_N_variables:        
        options['N_vars'] = N_variables
        
        # options['max_lag'] = max_lag
        
        for algorithm_paramters in algorithms_parameters.values():
            algorithm_paramters['max_lag'] = options['max_lag']
        
        yield algorithms_parameters, options
        
def changing_preselection_alpha(
    options: dict[str, Any],
    algorithms_parameters: dict[str, Any],
    list_preselection_alpha: Union[list[float], None]
) -> Generator[tuple[dict[str, Any], dict[str, Any]], None, None]:
    if list_preselection_alpha is None:
        list_preselection_alpha = [0.01, 0.05, 0.1, 0.2]
        
    for preselection_alpha in list_preselection_alpha:
        algorithms_parameters['pcmci-modified']['preselection_alpha'] = preselection_alpha
        
        yield algorithms_parameters, options

def changing_N_groups(
    options: dict[str, Any],
    algorithms_parameters: dict[str, Any],
    list_N_groups: Union[list[int], None] = None,
    relation_vars_per_group: int = 5
) -> Generator[tuple[dict[str, Any], dict[str, Any]], None, None]:
    if list_N_groups is None:
        list_N_groups = [5, 10, 20, 50]
    
    for N_groups in list_N_groups:
        options['N_groups'] = N_groups
        options['N_vars'] = N_groups * relation_vars_per_group

        yield algorithms_parameters, options

def changing_N_vars_per_group(
    options: dict[str, Any],
    algorithms_parameters: dict[str, Any],
    list_N_vars_per_group: Union[list[int], None] = None
) -> Generator[tuple[dict[str, Any], dict[str, Any]], None, None]:
    if list_N_vars_per_group is None:
        list_N_vars_per_group = [2, 4, 6, 8, 10, 12]
    
    for N_vars_per_group in list_N_vars_per_group:
        options['N_vars_per_group'] = N_vars_per_group
        options['N_vars'] = options['N_groups'] * N_vars_per_group
        
        for algorithm_parameters in algorithms_parameters.values():
            algorithm_parameters['max_lag'] = options['max_lag']
        
        yield algorithms_parameters, options

def increasing_N_vars_per_group(
    options: dict[str, Any],
    algorithms_parameters: dict[str, Any],
    list_N_vars_per_group: Union[list[int], None] = None
) -> Generator[tuple[dict[str, Any], dict[str, Any]], None, None]:
    if list_N_vars_per_group is None:
        list_N_vars_per_group = [2, 4, 6, 8, 10, 12]
    
    for N_vars_per_group in list_N_vars_per_group:
        if N_vars_per_group <= 6:
            algorithms_parameters['group-embedding']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.6
            algorithms_parameters['subgroups']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.6
        elif N_vars_per_group < 10:
            algorithms_parameters['group-embedding']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.5
            algorithms_parameters['subgroups']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.5
        elif N_vars_per_group < 12:
            algorithms_parameters['group-embedding']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.4
            algorithms_parameters['subgroups']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.4
        else:
            algorithms_parameters['group-embedding']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.3
            algorithms_parameters['subgroups']['dimensionality_reduction_params']['explained_variance_threshold'] = 0.3
        
        options['N_vars_per_group'] = N_vars_per_group
        options['N_vars'] = options['N_groups'] * N_vars_per_group
        
        for algorithm_parameters in algorithms_parameters.values():
            algorithm_parameters['max_lag'] = options['max_lag']
        
        yield algorithms_parameters, options

def changing_alg_params(
    options: dict[str, Any],
    algorithms_parameters: dict[str, Any],
    alg_name: str,
    list_modifying_algorithms_params: list[dict[str, Any]]
) -> Generator[tuple[dict[str, Any], dict[str, Any]], None, None]:
    for modifying_algorithm_params in list_modifying_algorithms_params:
        for param_name, param_value in modifying_algorithm_params.items():
            algorithms_parameters[alg_name][param_name] = param_value
        
        yield algorithms_parameters, options

'''
    EVALUATION METRICS
'''
ParentRef = Union[int, tuple[int, int]]
ParentGraph = Mapping[int, Sequence[ParentRef]]


def get_TP(ground_truth_parents: ParentGraph, predicted_parents: ParentGraph) -> int:
    # TP = |{predicted_parents} ∩ {ground_truth_parents}|
    true_positives = 0
    for effect, causes in predicted_parents.items():
        true_positives += len([cause for cause in causes if cause in ground_truth_parents.get(effect, [])])
    
    return true_positives

def get_FP(ground_truth_parents: ParentGraph, predicted_parents: ParentGraph) -> int:
    # FP = |{predicted_parents} - {ground_truth_parents}|
    false_positives = 0
    for effect, causes in predicted_parents.items():
        false_positives += len([cause for cause in causes if cause not in ground_truth_parents.get(effect, [])])
    
    return false_positives

def get_FN(ground_truth_parents: ParentGraph, predicted_parents: ParentGraph) -> int:
    # FN = |{ground_truth_parents} - {predicted_parents}|
    false_negatives = 0
    for effect, causes in ground_truth_parents.items():
        false_negatives += len([cause for cause in causes if cause not in predicted_parents.get(effect, [])])
    
    return false_negatives

def get_precision(ground_truth_parents: ParentGraph, predicted_parents: ParentGraph) -> float:
    # Precision = TP / (TP + FP)
    true_positives = get_TP(ground_truth_parents, predicted_parents)
    
    false_positives = get_FP(ground_truth_parents, predicted_parents)
    
    denominator = true_positives + false_positives
    return true_positives / denominator if denominator != 0 else 0

def get_recall(ground_truth_parents: ParentGraph, predicted_parents: ParentGraph) -> float:
    # Recall = TP / (TP + FN)
    true_positives = get_TP(ground_truth_parents, predicted_parents)
    
    false_negatives = get_FN(ground_truth_parents, predicted_parents)
    
    denominator = true_positives + false_negatives
    return true_positives / denominator if denominator != 0 else 0

def get_f1(ground_truth_parents: ParentGraph, predicted_parents: ParentGraph) -> float:
    precision = get_precision(ground_truth_parents, predicted_parents)
    recall = get_recall(ground_truth_parents, predicted_parents)
    
    return 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

def get_false_positive_ration(ground_truth_parents: ParentGraph, predicted_parents: ParentGraph) -> float:
    # FPR = FP / (FP + TN)
    false_positives = 0
    for effect, causes in predicted_parents.items():
        false_positives += len([cause for cause in causes if cause not in ground_truth_parents.get(effect, [])])
    
    true_negatives = 0
    for effect, causes in ground_truth_parents.items():
        true_negatives += len([cause for cause in causes if cause not in predicted_parents.get(effect, [])])
    
    return false_positives / (false_positives + true_negatives) if false_positives != 0 else 0

def get_shd(graph1: ParentGraph, graph2: ParentGraph) -> int:
    """Calculate the Structural Hamming Distance between two graphs."""
    def dict_to_adjacency_matrix(graph_dict, nodes):
        """Convert a graph dictionary to an adjacency matrix."""
        size = len(nodes)
        index = {node: i for i, node in enumerate(nodes)}
        adj_matrix = np.zeros((size, size), dtype=int)
        for child, parents in graph_dict.items():
            for parent in parents:
                adj_matrix[index[parent], index[child]] = 1
        return adj_matrix
    
    nodes = set(graph1.keys()).union(set(graph2.keys()))
    node_refs: set[ParentRef] = set(nodes)
    for parents in graph1.values():
        node_refs.update(parents)
    for parents in graph2.values():
        node_refs.update(parents)
    nodes = list(node_refs)
    
    adj_matrix1 = dict_to_adjacency_matrix(graph1, nodes)
    adj_matrix2 = dict_to_adjacency_matrix(graph2, nodes)
    
    # Calculate the number of differing edges
    diff = np.abs(adj_matrix1 - adj_matrix2)
    shd = np.sum(diff)
    
    return shd



'''
TIME SERIES GRAPHS UTILITIES
'''
def window_to_summary_graph(window_graph: dict[int, list[tuple[int, int]]]) -> dict[int, list[tuple[int, int]]]:
    '''
    Convert a window graph, in the way X^i_t' -> X^j_t
        to a summary graph, X^i_- ->X^j_t
    
    Args:
        window_graph : dict[int, list[tuple[int, int]]]
            A dictionary where the keys are the time points and the values are lists of parents.
            Each parent is a tuple (node, lag).
    
    Returns:
        summary_graph : dict[int, list[int]]
            A dictionary where the keys are the time points and the values are lists of parents.
            Each parent is a node.
    '''
    summary_graph = {}
    for t, parents in window_graph.items():
        summary_graph[t] = [(parent, -1) for parent, lag in parents if lag < 0]
        summary_graph[t] += [(parent, 0) for parent, lag in parents if lag == 0]
        # Remove duplicates
        summary_graph[t] = list(set(summary_graph[t]))
        
    return summary_graph
