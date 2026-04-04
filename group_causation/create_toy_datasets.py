'''
Module with the different functions that are necessary to generate toy time series datasets
from causal processes, which are defined by ts DAGs.
'''



from collections import deque
import os
import random
from typing import Callable, Union
import warnings
import numpy as np
import pandas as pd
from tigramite.toymodels.structural_causal_processes import generate_structural_causal_process, structural_causal_process
from tigramite import plotting as tp
from tigramite.graphs import Graphs


def get_parents_dict(causal_process) -> dict[int, list[tuple[int, int]]]:
    parents_dict = dict()
    for key in causal_process.keys():
        if key not in parents_dict:
            parents_dict[key] = []
        for i in range(len(causal_process[key])):
            parents_dict[key].append(causal_process[key][i][0])
    return parents_dict

class CausalDataset:
    def __init__(self, time_series=None, parents_dict=None, groups=None, max_value_threshold=1e10):
        '''
        Initialize the CausalDataset object.
        
        Args:
            time_series : np.ndarray with shape (n_samples, n_variables)
            parents_dict : dictionary whose keys are each node, and values are the lists of parents, [... (i, -tau) ...].
            groups : List of lists, where each list is a group of variables. Just is used in case of group-based datasets.
            max_value_threshold : Maximum value of the time series. If a value is greater than this, generation will be repeated.
        '''
        self.time_series: Union[np.ndarray, None] = time_series
        self.parents_dict: Union[dict[int, list[tuple[int, int]]], None] = parents_dict
        self._groups: Union[list[list[int]], None] = groups
        self.node_parents_dict: dict[int, list[tuple[int, int]]] = {}
        self.max_value_threshold = max_value_threshold

    @property
    def groups(self) -> Union[list[list[int]], None]:
        return self._groups

    @groups.setter
    def groups(self, value: Union[list[list[int]], None]) -> None:
        self._groups = value
    
    dependency_funcs_dict = {
        'linear': lambda x: x,
        'negative-exponential': lambda x: 1 - np.exp(-abs(x)),
        'sin': lambda x: np.sin(x),
        'cos': lambda x: np.cos(x),
        'step': lambda x: 1 if x > 0 else -1,
    }
    
    def generate_toy_data(self, name, T=100, N_vars=10, crosslinks_density=0.75,
                      confounders_density = 0, min_lag=1, max_lag=3, contemp_fraction=0,
                      dependency_funcs=['nonlinear'], datasets_folder = None, maximum_tries=100,
                      **kw_generation_args) \
                            -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
        r"""
        Generate a toy dataset with a causal process and time series data.
        Node-level links are modeled as a linear combination of the parents, in the following way:
        
        .. math:: X_i^t = \sum_{j \in \\text{parents}(i)} \\beta_{ij} X_j^{t - \\tau_{ij}} + \epsilon_i(t)
        
        Where the scalar coefficients are takien from kw_generation_args 'dependency_coeffs' and 'auto_coeffs' parameters,
        and the noise :math:`\epsilon_i(t)` is taken from kw_generation_args 'noise_dists' and 'noise_sigmas' parameters.
        
        Args:
            name : Name of the dataset
            T : Number of time points
            N : Number of variables
            crosslinks_density : Fraction of links that are cross-links
            confounders_density : Fraction of confounders in the dataset
            max_lag : Maximum lag of the causal process
            dependency_funcs : List of dependency functions (in {'linear', 'nonlinear'}, or a function :math:`f:\mathbb R \\rightarrow\mathbb R`)
            dataset_folder : Name of the folder where datasets and parents will be saved. By default they are not saved.
            
        Returns:
            time_series : np.ndarray with shape (n_samples, n_variables)
            parents_dict: dictionary whose keys are each node, and values are the lists of parents, [... (i, -tau) ...].
        """
        if min_lag > 0 and contemp_fraction > 1e-6:
            raise ValueError('If min_lag > 0, then contemp_fraction must be 0')
        elif min_lag == 0 and contemp_fraction < 1e-6:
            raise ValueError('If min_lag is 0, then contemp_fraction can not be 0.')
        
        # Convert dependency_funcs names to functions
        dependency_funcs = [self.dependency_funcs_dict[func] if func in self.dependency_funcs_dict else func
                                for func in dependency_funcs ]
        
        L = N_vars * crosslinks_density / (1 - crosslinks_density) # Forcing crosslinks_density = L / (N + L)
        L = int(L//(1-contemp_fraction)) # So that the contemp links are not counted in L
        total_generating_vars = int(N_vars * (1 + confounders_density))
        
        # Try to generate data until there are no NaNs
        for it in range(1, maximum_tries+1):
            # Generate random causal process
            causal_process, noise = generate_structural_causal_process(N=total_generating_vars,
                                                                L=L,
                                                                max_lag=max_lag,
                                                                contemp_fraction=contemp_fraction,
                                                                dependency_funcs=dependency_funcs,
                                                                **kw_generation_args)
            self.parents_dict = get_parents_dict(causal_process)
            # Generate time series data from the causal process
            self.time_series, _ = structural_causal_process(causal_process, T=T, noises=noise)
            if confounders_density > 1e-6:
                # Now we choose what variables will be kept and studied (the rest are hidden confounders)
                chosen_nodes = random.sample(range(total_generating_vars), N_vars)
                self.time_series = self.time_series[:, chosen_nodes]
                self.parents_dict = _extract_subgraph(self.parents_dict, chosen_nodes)
            # If dataset has no NaNs nor infinites, use it
            if np.all(np.isfinite(self.time_series)) and \
                np.all(np.abs(self.time_series) < self.max_value_threshold) and \
                not np.any(np.isnan(self.time_series)):
                break
            else:
                print(f'Dataset has NaNs or infinites, trying again... {it}/{maximum_tries}')
        
        # If the maximum number of tries is reached, raise an error
        if it == maximum_tries:
            raise ValueError('Current Could not generate a dataset without NaNs')
            
        if datasets_folder is not None:
            # If the folder does not exist, create it
            if not os.path.exists(datasets_folder):
                os.makedirs(datasets_folder)
            # Save the dataset
            self._save(name, datasets_folder)
                
        assert self.time_series is not None
        assert self.parents_dict is not None
        return self.time_series, self.parents_dict
    
    def _save(self, name, dataset_folder):
        # Save the time series data to a csv file
        df = pd.DataFrame(self.time_series)
        df.to_csv(f'{dataset_folder}/{name}_data.csv', index=False, header=True)
        # Save parents to a txt file
        with open(f'{dataset_folder}/{name}_parents.txt', 'w') as f:
            parents_representation = repr(self.parents_dict)
            f.write(parents_representation)
    
    def generate_group_toy_data(self, name, T=100, N_vars=20, N_groups=3,
                                inner_group_crosslinks_density=0.5, outer_group_crosslinks_density=0.5,
                                n_node_links_per_group_link=2, contemp_fraction=.0,
                                max_lag=3, min_lag=1, dependency_funcs=['nonlinear'],
                                dependency_coeffs=[-0.5, 0.5], auto_coeffs=[0.5, 0.7],
                                noise_dists=['gaussian'], noise_sigmas=[0.5, 2],
                                datasets_folder = None, maximum_tries=100, **kw_generation_args) \
                            -> tuple[np.ndarray, dict[int, list[tuple[int, int]]], list[list[int]], dict[int, list[tuple[int, int]]]]:
        r'''
        Generate a toy dataset with a group based causal process and time series data.
        There will first be generated N_groups different groups of variables, each with between 2 and 
        N_vars - 2*N_groups variables. Then, there will be generated links between the groups,
        with a density of outer_group_crosslinks_density. Finally, there will be generated links between
        nodes of different groups, with a density of inner_group_crosslinks_density.
        Node-level links are modeled as a linear combination of the parents, in the following way:
        
        .. math:: X_i^t = \sum_{j \in \\text{parents}(i)} \\beta_{ij} X_j^{t - \\tau_{ij}} + \epsilon_i(t)
        
        Where the scalar coefficients are takien from kw_generation_args 'dependency_coeffs' and 'auto_coeffs' parameters,
        and the noise :math:`\epsilon_i(t)` is taken from kw_generation_args 'noise_dists' and 'noise_sigmas' parameters.
        
        Args:
            name : Name of the dataset
            T : Number of time points
            N : Number of variables
            N_groups : Number of groups (this number must be greater than :math:`N/2`, to allow groups creation)
            inner_group_crosslinks_density : Percentage of inner-group links that are cross-links
            n_node_links_per_group_link : Number of links between nodes of different groups that are linked
            max_lag : Maximum lag of the causal process
            dependency_funcs : List of dependency functions (in ['linear', 'nonlinear'], or a function :math:`f:\mathbb R \\rightarrow\mathbb R`)
            dataset_folder : Name of the folder where datasets and parents will be saved. By default they are not saved.
            
        Returns:
            time_series : np.ndarray with shape (n_samples, n_variables).
            group_parents_dict : dictionary whose keys are each group, and values are the lists of parent groups, [... (i, -tau) ...].
            groups : List of lists, where each list is a group of variables
            node_parents_dict : dictionary whose keys are each node, and values are the lists of parents, [... (i, -tau) ...].
        '''
        # Check that parameters are consistent
        if min_lag > 0 and contemp_fraction > 1e-6:
            raise ValueError('If there is a fraction of links that are contemporaneous, the minimum lag must be 0')
        # Convert dependency_funcs names to functions
        dependency_funcs = [self.dependency_funcs_dict[func] if func in self.dependency_funcs_dict else func\
                                for func in dependency_funcs]
        
        # Try to generate data until there are no NaNs
        for it in range(1, maximum_tries+1):
            self._groups = self._generate_groups(N_vars, N_groups)
            
            # Dictionary where keys will be the global index of the nodes, and values the causal processes
            groups_causal_processes = dict()
            
            # Generate inner causal processes
            for index, group in enumerate(self._groups):
                # Forcing crosslinks_density = L / (N + L)
                L = int(len(group) * inner_group_crosslinks_density / (1 - inner_group_crosslinks_density))
                L = int(L * (1+contemp_fraction))
                causal_process, _ = generate_structural_causal_process(N=len(group), L=L, max_lag=max_lag,
                                                                        dependency_funcs=dependency_funcs,
                                                                        contemp_fraction=contemp_fraction,
                                                                        dependency_coeffs=dependency_coeffs,
                                                                        auto_coeffs=auto_coeffs,)
                
                # Change the keys of the causal process to the global index
                groups_causal_processes[index] = self._change_keys(causal_process, group)
            
            # Generate outer causal processes
            L = int(N_groups * outer_group_crosslinks_density / (1 - outer_group_crosslinks_density))
            L = int(L * (1+contemp_fraction))
            outer_causal_process, _ = generate_structural_causal_process(N=N_groups, L=L, max_lag=max_lag,
                                                                        dependency_funcs=dependency_funcs,
                                                                        contemp_fraction=contemp_fraction,
                                                                        dependency_coeffs=dependency_coeffs,
                                                                        auto_coeffs=auto_coeffs,)
            
            global_causal_process = self._join_processes( outer_causal_process, groups_causal_processes,
                                                        n_node_links_per_group_link)
            
            # Generate noise
            _, noise = generate_structural_causal_process(N=N_vars, noise_dists=noise_dists,
                                                          noise_sigmas=noise_sigmas)
            
            self.node_parents_dict = get_parents_dict(global_causal_process)
            self.parents_dict = self.extract_group_parents(self.node_parents_dict)
            
            # Generate time series data from the causal process
            self.time_series, _ = structural_causal_process(global_causal_process, T=T, noises=noise)

            # If dataset has no NaNs nor infinites, use it
            if np.all(np.isfinite(self.time_series)) and \
                np.all(np.abs(self.time_series) < self.max_value_threshold) and \
                not np.any(np.isnan(self.time_series)):
                break
            else:
                print(f'Dataset has NaNs or infinites, trying again... {it}/{maximum_tries}')
        
        # If the maximum number of tries is reached, raise an error
        if it == maximum_tries:
            raise ValueError('Could not generate a dataset without NaNs.')
            
        if datasets_folder is not None:
            # If the folder does not exist, create it
            if not os.path.exists(datasets_folder):
                os.makedirs(datasets_folder)
            # Save the dataset
            self._save_groups(name, datasets_folder)
        
        assert self.time_series is not None
        assert self.parents_dict is not None
        assert self._groups is not None
        return self.time_series, self.parents_dict, self._groups, self.node_parents_dict
    
    def extract_group_parents(self, node_parents_dict: dict[int, list[tuple[int, int]]]) -> dict[int, list[tuple[int, int]]]:
        '''
        Given a dictionary with the parents of each node, return a dictionary with the parents of each group.
        
        Args:
            node_parents_dict : dictionary whose keys are each node, and values are the lists of parents, [... (i_node, -tau) ...].
            
        Returns:
            group_parents_dict: dictionary whose keys are each group, and values are the lists of parent groups, [... (i_group, -tau) ...].
        '''
        assert self._groups is not None
        group_parents_dict: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(self._groups))}
        
        # Iterate over the nodes and their parents
        for son_node, parents in node_parents_dict.items():
            [son_group] = [i for i, group in enumerate(self._groups) if son_node in group]
            for parent, lag in parents:
                [parent_group] = [i for i, group in enumerate(self._groups) if parent in group]
                # Add the parent group to the son group
                group_parents_dict[son_group].append((parent_group, lag))
            
            # Remove duplicates
            group_parents_dict[son_group] = list(set(group_parents_dict[son_group]))
            # Remove autolinks
            for son, parents in group_parents_dict.items():
                group_parents_dict[son] = [(parent,lag) for (parent,lag) in parents\
                                                if parent!=son or lag!=0]
        
        return group_parents_dict
        
    def _generate_groups(self, N_vars, N_groups) -> list[list[int]]:
        '''
        Generate N_groups groups of variables, with at least 2 nodes each.
        
        Args:
            N_vars : Number of variables
            N_groups : Number of groups
        
        Returns:
            groups : List of lists, where each list is a group of variables
        '''
        if N_groups > N_vars/2:
            raise ValueError('The number of groups must be less than N_vars / 2')
        
        # Generate N_groups groups with 2 nodes each
        nodes = list( range(N_vars) )
        groups = [[nodes.pop(), nodes.pop()] for _ in range(N_groups)]
        
        # Distribute the remaining nodes randomly
        while len(nodes) != 0:
            group = groups[np.random.randint(0, N_groups)]
            group.append(nodes.pop())

        return groups
    
    def _change_keys(self, causal_process, group) -> dict[int, list[tuple[tuple, int, Callable]]]:
        '''
        Change the keys of the causal process to the global index, through a bijective mapping
        
        
        Args:
            causal_process : dictionary whose keys are the local index of the nodes, and values are the lists of tuples
                                [ ((parent, lag), coeff, func), ... ]
            group : list of the global index of the nodes
            
        Returns:
            new_causal_process : dictionary whose keys are the global index of the nodes, and values are the lists of tuples
                                [ ((parent, lag), coeff, func), ... ]
        '''
        # Dictionary where keys will be original key, and values their associated node
        self.association_relation = dict( zip(causal_process.keys(), group) )
        new_causal_process = dict()
        
        for original_key, new_index in self.association_relation.items():
            original_parents = causal_process[original_key]
            
            # We need to change the parents to the global index
            change_parent_index = lambda parent, lag: (self.association_relation[parent], lag)
            new_parents = [(change_parent_index(*node), coeff, func) for node, coeff, func in original_parents]
 
            new_causal_process[new_index] = new_parents
        
        return new_causal_process
            
    def _join_processes(self, outer_causal_process: dict[int, list[tuple[tuple[int, int], float, Callable]]],
                            groups_causal_processes: dict[int, dict[int, list[tuple[tuple[int, int], float, Callable]]]],
                            n_node_links_per_group_link) -> dict[int, list[tuple[tuple, int, Callable]]]:
        '''
        Join several causal processes into one. The outer_causal_process will be converted
        to the global index by assigning n_node_links_per_group_link links between nodes
        of the different groups.
        
        Args:
            outer_causal_process : dictionary whose keys are the index of the groups, and values are the lists of tuples
                                [ ((parent_group, lag), coeff, func), ... ]
            groups_causal_processes : dictionaries whose keys are the groups index, and whose items are dictionaries
                                with the same structure as outer_causal_process, i.e., keys are global index of the nodes,
                                and values are the lists of tuples [ ((parent, lag), coeff, func), ... ]
            n_node_links_per_group_link : Number of links between nodes of different groups that are linked
        
        Returns:
            global_causal_process : dictionary whose keys are the global index of the nodes, and values are the lists of tuples
                                [ ((parent, lag), coeff, func), ... ]
        '''
        assert self._groups is not None
        global_causal_process = {i: [] for group in self._groups for i in group}
        
        # Assign the outer links
        for son_group, group_parents in outer_causal_process.items():
            for (parent_group_index, lag), coeff, func in group_parents:
                for _ in range(n_node_links_per_group_link):
                    # Choose a random node from each group
                    origin_node = random.choice(self._groups[son_group])
                    # Choose a random node from the parent group
                    parent_node = random.choice(self._groups[parent_group_index])
                    # Assign the link
                    global_causal_process[origin_node].append( ((parent_node, lag), coeff, func) )
        
        # Assign the inner links
        for son_group, causal_process in groups_causal_processes.items():
            for son_node, parents in causal_process.items():
                global_causal_process[son_node].extend(parents)
        
        # Delete duplicates
        for key, parents in global_causal_process.items():
            seen_parents_nodes = set()
            seen_parents_processes = []
            for parent in parents:
                if parent[0] in seen_parents_nodes:
                    parents.remove(parent)
                else:
                    seen_parents_nodes.add(parent[0])
                    seen_parents_processes.append(parent)
            # Update the global_causal_process with the new list of non-repeated parents
            global_causal_process[key] = seen_parents_processes
        
        return global_causal_process
    
    def _save_groups(self, name, dataset_folder):
        self._save(name, dataset_folder)
        # Save the groups to a txt file
        with open(f'{dataset_folder}/{name}_groups.txt', 'w') as f:
            groups_representation = repr(self._groups)
            f.write(groups_representation)
        # Save the groups parents to a txt file
        with open(f'{dataset_folder}/{name}_node_parents.txt', 'w') as f:
            node_parents_representation = repr(self.node_parents_dict)
            f.write(node_parents_representation)
    

def plot_ts_graph(parents_dict, var_names=None):
    '''
    Function to plot the graph structure of the time series
    '''
    graph = Graphs.get_graph_from_dict(parents_dict)
    tp.plot_time_series_graph(
        graph=graph,
        var_names=var_names,
        link_colorbar_label='cross-MCI (edges)',
    )


def _extract_subgraph(parents: dict[int, list[tuple[int,int]]],
                     chosen_nodes: list[int]
                    ) -> dict[int, list[tuple[int,int]]]:
    '''
    Given a dictionary with the parents of each node in a graph,
    return a dictionary with the parents of chosen nodes, considering that
    a variable between the chosen_nodes is son of another if and only if
    there is a directed path from the parent to the child that only goes
    through non-chosen nodes.
    '''    
    chosen_set = set(chosen_nodes)
    idx_of = {node: i for i, node in enumerate(chosen_nodes)}
    
    new_parents = {idx: [] for idx in idx_of.values()}
    
    for child in chosen_nodes:
        child_idx = idx_of[child]
        
        # BFS queue entries are (current_node, cum_lag)
        queue = deque([(child, 0)])
        visited = {child}
        
        while queue:
            curr, cum_lag = queue.popleft()
            
            for p, lag in parents.get(curr, []):
                if p in visited:
                    continue
                total_lag = cum_lag + lag
                
                if p in chosen_set:
                    # avoid trivial self-loop with lag==0
                    if not (p == child and lag == 0):
                        new_parents[child_idx].append((idx_of[p], total_lag))
                    # do *not* walk past a chosen node
                else:
                    visited.add(p)
                    queue.append((p, total_lag))
        
        # remove duplicates (in case multiple paths hit the same chosen parent)
        seen = set()
        uniq = []
        for pair in new_parents[child_idx]:
            if pair not in seen:
                seen.add(pair)
                uniq.append(pair)
        new_parents[child_idx] = uniq
    
    return new_parents             


if __name__ == '__main__':
    # dataset = CausalDataset()
    # # dataset.generate_toy_data('1')
    
    # ts, node_parents_dict, group_parents_dict, groups = dataset.generate_group_toy_data('1',
    #                                                                                     N_vars=6,
    #                                                                                     N_groups=3,)
    
    # _plot_ts_graph(node_parents_dict)
    # _plot_ts_graph(group_parents_dict)
    # plt.show()
    
    
    # Test _extract_subgraph
    parents_dict = {0: [(0, -1)], 1: [(1, -1), (3, -4), (6, 0)], 2: [(2, -1)], 3: [(3, -1), (1, -2), (6, -2)], 4: [(4, -1)], 5: [(5, -1), (9, -2), (0, -2)], 6: [(6, -1), (0, -1)], 7: [(7, -1), (3, -2), (8, 0)], 8: [(8, -1)], 9: [(9, -1), (1, -5)]}
    chosen_nodes = [8, 0, 2, 7, 5, 9, 4, 3, 1, 6]
    print(_extract_subgraph(parents_dict, chosen_nodes))