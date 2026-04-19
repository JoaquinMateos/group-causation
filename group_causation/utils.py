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
import causaldag as cd
from typing import Any, Tuple, Set, Union, List, Dict

ParentRef = Tuple[int, int]
ParentGraph = Mapping[int, Sequence[ParentRef]]

def get_cpdag_and_edge_set(graph_dict: ParentGraph) -> Tuple[Set[Any], cd.PDAG]:
    """
    Helper function to convert a parents dictionary into a causaldag PDAG.
    This function must be applied only on the contemporaneous graph,
    where all lags have been collapsed to 0, to ensure correct MEC evaluation.
        Logic: Inside the MEC calculation, we collapse all nodes to integers 
        to ensure the library identifies v-structures correctly.
    """
    dag = cd.DAG()
    
    # 1. Collect all unique nodes as INTEGERS
    nodes: Set[int] = set()
    nodes.update(graph_dict.keys())
    for parents in graph_dict.values():
        # Unpack the node from the (node, lag) tuple
        nodes.update([p[0] for p in parents])
        
    for node in nodes:
        dag.add_node(node)
        
    # 2. Build the directed edges using INTEGERS
    for child, parents in graph_dict.items():
        for parent_ref in parents:
            # We can safely extract the parent node as an integer because ParentRef is strictly Tuple[int, int]
            parent_node = int(parent_ref[0]) if isinstance(parent_ref[0], (int, float)) else parent_ref[0]
            try:
                # Direct integer-to-integer arc
                dag.add_arc(parent_node, child)
            except Exception:
                pass 
                
    # 3. Convert to MEC
    cpdag = dag.cpdag()
    
    # 4. Create a canonical set of edge statuses 
    edge_set = set()
    for u, v in cpdag.arcs:
        edge_set.add(('directed', u, v))
    for u, v in cpdag.edges:
        edge_set.add(('undirected', frozenset([u, v])))
        
    return edge_set, cpdag

def split_lagged_and_contemporaneous(graph_dict: ParentGraph) -> Tuple[ParentGraph, ParentGraph]:
    """Splits a window graph into lagged (keeps tuple) and contemporaneous (strips lag to integer) subgraphs."""
    lagged_graph: Dict[int, List[ParentRef]] = {node: [] for node in graph_dict.keys()}
    contemp_graph: Dict[int, List[ParentRef]] = {node: [] for node in graph_dict.keys()}

    for child, parents in graph_dict.items():
        for parent_ref in parents:
            # We can now safely unpack directly because ParentRef is strictly Tuple[int, int]
            parent_node, lag = parent_ref
            
            if lag < 0:
                lagged_graph[child].append(parent_ref) # Keep full tuple for strict DAG matching
            else:
                contemp_graph[child].append((int(parent_node), 0)) # Strip lag for CPDAG

    return lagged_graph, contemp_graph

def get_dag_edge_set(graph_dict: ParentGraph) -> Set[Any]:
    """Returns a canonical set of strictly directed edges for exact DAG evaluation."""
    edge_set = set()
    for child, parents in graph_dict.items():
        for parent in parents:
            edge_set.add(('directed', parent, child))
    return edge_set


# --- METRIC FUNCTIONS ---
def get_TP(gt_edges: set, pred_edges: set) -> int:
    return len(gt_edges.intersection(pred_edges))

def get_FP(gt_edges: set, pred_edges: set) -> int:
    return len(pred_edges - gt_edges)

def get_FN(gt_edges: set, pred_edges: set) -> int:
    return len(gt_edges - pred_edges)

def get_precision(gt_edges: set, pred_edges: set) -> float:
    tp = get_TP(gt_edges, pred_edges)
    fp = get_FP(gt_edges, pred_edges)
    denominator = tp + fp
    return tp / denominator if denominator != 0 else 0

def get_recall(gt_edges: set, pred_edges: set) -> float:
    tp = get_TP(gt_edges, pred_edges)
    fn = get_FN(gt_edges, pred_edges)
    denominator = tp + fn
    return tp / denominator if denominator != 0 else 0

def get_f1(gt_edges: set, pred_edges: set) -> float:
    precision = get_precision(gt_edges, pred_edges)
    recall = get_recall(gt_edges, pred_edges)
    return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

def get_false_positive_ratio(gt_edges: set, pred_edges: set, n_nodes: int) -> float:
    fp = get_FP(gt_edges, pred_edges)
    
    total_possible_edges = (n_nodes * (n_nodes - 1)) / 2
    actual_positives = len(gt_edges)
    actual_negatives = total_possible_edges - actual_positives
    
    return fp / actual_negatives if actual_negatives > 0 else 0

def get_shd(gt_cpdag: cd.PDAG, pred_cpdag: cd.PDAG) -> int:
    """Calculate the Structural Hamming Distance between two PDAGs."""
    return gt_cpdag.shd(pred_cpdag)


def get_global_window_metrics(
    gt_lagged_edges: set, pred_lagged_edges: set,
    gt_contemp_edges: set, pred_contemp_edges: set,
    gt_contemp_cpdag: cd.PDAG, pred_contemp_cpdag: cd.PDAG
) -> dict[str, float]:
    """
    Calculates the combined global window metrics (F1, Precision, Recall, SHD) 
    across both lagged and contemporaneous subgraphs.
    """
    total_tp = get_TP(gt_lagged_edges, pred_lagged_edges) + get_TP(gt_contemp_edges, pred_contemp_edges)
    total_fp = get_FP(gt_lagged_edges, pred_lagged_edges) + get_FP(gt_contemp_edges, pred_contemp_edges)
    total_fn = get_FN(gt_lagged_edges, pred_lagged_edges) + get_FN(gt_contemp_edges, pred_contemp_edges)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Lagged SHD is strictly FP + FN (no orientation errors exist)
    shd_lagged = get_FP(gt_lagged_edges, pred_lagged_edges) + get_FN(gt_lagged_edges, pred_lagged_edges)
    shd_contemp = get_shd(gt_contemp_cpdag, pred_contemp_cpdag)
    shd_global = shd_lagged + shd_contemp
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'shd': shd_global
    }


'''
TIME SERIES GRAPHS UTILITIES
'''
def window_to_summary_graph(window_graph: dict[int, list[tuple[int, int]]]) -> dict[int, list[tuple[int, int]]]:
    '''
    Convert a window graph, in the way X^i_t' -> X^j_t
        to a summary graph, X^i_- ->X^j_t
    Maintains strict (node, lag) format but collapses all non-zero lags to -1.
    
    Args:
        window_graph : dict[int, list[tuple[int, int]]]
            A dictionary where the keys are the time points and the values are lists of parents.
            Each parent is a tuple (node, lag).
    
    Returns:
        summary_graph : dict[int, list[tuple[int, int]]]
            A dictionary where the keys are the time points and the values are lists of parents.
            Each parent is a tuple (node, lag).
    '''
    summary_graph = {node: [] for node in window_graph.keys()}
    
    for son, parents in window_graph.items():
        for parent_info in parents:
            # Extraer solo el ID del padre (si viene en formato tupla)
            parent_node = parent_info[0] if isinstance(parent_info, tuple) else parent_info
            
            # Condición para evitar auto-bucles (diagonal) y duplicados
            if parent_node != son and parent_node not in summary_graph[son]:
                summary_graph[son].append(parent_node)
                
    return summary_graph
