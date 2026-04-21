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
from group_causation.data_management.time_series_generator import generate_group_causal_process_structure, generate_data_from_causal_process_structure
from tigramite import plotting as tp
from tigramite.graphs import Graphs


def get_parents_dict(causal_process) -> dict[int, list[tuple[int, int]]]:
    '''
    Extracts parents dict. Supports both Tigramite's format and the Multivariate format.
    '''
    parents_dict = {}
    for key, links in causal_process.items():
        parents_dict[key] = []
        for link in links:
            parents_info = link[0]
            # If it's the multivariate format (tuple of tuples)
            if isinstance(parents_info[0], tuple):
                for p, lag in parents_info:
                    if (p, lag) not in parents_dict[key]:
                        parents_dict[key].append((p, lag))
            # If it's the Tigramite format ((parent, lag), coeff, func)
            else: 
                p, lag = parents_info
                if (p, lag) not in parents_dict[key]:
                    parents_dict[key].append((p, lag))
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
                                latent_confounding_fraction=0.0,
                                n_node_links_per_group_link=2, contemp_fraction=.0,
                                cross_terms_fraction=0.2,
                                max_lag=3, min_lag=1, dependency_funcs=['linear'],
                                multivariate_funcs=[lambda x, y: x * y],
                                dependency_coeffs=[-0.5, 0.5], auto_coeffs=[0.5, 0.7],
                                noise_dists=['gaussian'], noise_sigmas=[0.5, 2],
                                datasets_folder = None, maximum_tries=100, 
                                group_links = None,
                                **kw_generation_args) \
                            -> tuple[np.ndarray, dict[int, list[tuple[int, int]]], list[list[int]], dict[int, list[tuple[int, int]]]]:
        '''
        Generate a toy dataset with a group-based causal process and time series data.
        
        Args:
            name : Name of the dataset
            T : Number of time points
            N_vars : Number of variables
            N_groups : Number of groups
            inner_group_crosslinks_density : Density of links between nodes of the same group
            outer_group_crosslinks_density : Density of links between groups
            latent_confounding_fraction : Fraction of latent confounders at the group level (these are groups that are generated but then hidden, so they create latent confounding between the visible groups)
            n_node_links_per_group_link : Number of node-level links per group-level link
            contemp_fraction : Fraction of links that are contemporaneous (lag 0)
            cross_terms_fraction : Fraction of links that are cross-terms (multivariate interactions from multiple parents, instead of simple univariate functions of each parent)
            max_lag : Maximum lag of the causal process
            min_lag : Minimum lag of the causal process (if 0, there can be contemporaneous links)
            dependency_funcs : List of dependency functions (in {'linear', 'negative-exponential', 'sin', 'cos', 'step'}, or a function :math:`f:\mathbb R \\rightarrow\mathbb R`)
            multivariate_funcs : List of multivariate functions (functions :math:`f:\mathbb R^k \\rightarrow\mathbb R`, where k is the number of parents in the interaction)
            dependency_coeffs : List of coefficients for the parent dependencies (these are the :math:`\\beta_{ij}` in the equation in the docstring of generate_toy_data)
            auto_coeffs : List of coefficients for the auto-dependencies (lags of the same variable)
            noise_dists : List of noise distributions for each variable (in {'gaussian'}, or a function that generates noise given the number of samples)
            noise_sigmas : List of noise standard deviations for each variable (if noise_dists is 'gaussian', these are the standard deviations of the Gaussian noise)
            datasets_folder : Name of the folder where datasets and parents will be saved. By default they are not saved.
            group_links : Optional dictionary defining the Macro-Graph structure. If None, it will be generated randomly using outer_group_crosslinks_density and contemp_fraction
        '''
        if min_lag > 0 and contemp_fraction > 1e-6:
            raise ValueError('If there is a fraction of links that are contemporaneous, the minimum lag must be 0')
        
        # Convert dependency_funcs names to functions
        parsed_dependency_funcs = [self.dependency_funcs_dict[func] if func in self.dependency_funcs_dict else func\
                                for func in dependency_funcs]
        
        # Calculate inflated geometry for latent groups
        total_groups = int(N_groups * (1 + latent_confounding_fraction))
        total_vars = int(N_vars * (1 + latent_confounding_fraction))
        
        for it in range(1, maximum_tries+1):
            try:
                # 1. Set Groups
                current_groups = self._generate_groups(total_vars, total_groups)
                
                # 2. Set the Macro-Graph (Group Links)
                current_group_links = group_links
                if current_group_links is None:
                    current_group_links = self._generate_random_group_links(
                        N_groups=total_groups, # Use total_groups
                        density=outer_group_crosslinks_density, 
                        max_lag=max_lag, 
                        contemp_fraction=contemp_fraction
                    )
                
                # 3. Generate the Micro-Graph structure
                global_causal_process = generate_group_causal_process_structure(
                    groups=current_groups,
                    group_links=current_group_links,
                    n_node_links_per_group_link=n_node_links_per_group_link,
                    inner_group_density=inner_group_crosslinks_density,
                    max_lag=max_lag,
                    contemp_fraction=contemp_fraction,
                    cross_terms_fraction=cross_terms_fraction,
                    dependency_funcs=parsed_dependency_funcs,
                    multivariate_funcs=multivariate_funcs,
                    dependency_coeffs=dependency_coeffs,
                    auto_coeffs=auto_coeffs,
                    enforce_stationarity=True 
                )
                
                # 4. Generate full inflated data
                full_time_series, nonvalid = generate_data_from_causal_process_structure(
                    links=global_causal_process,
                    T=T,
                    noise_dists=noise_dists,
                    noise_sigmas=noise_sigmas
                )
                
                if nonvalid or np.any(np.abs(full_time_series) > self.max_value_threshold):
                    print(f'Dataset has NaNs or infinites, trying again... {it}/{maximum_tries}')
                    continue
                
                # 5. Extract Visible Subgraph & Apply Latent Hiding
                if latent_confounding_fraction > 1e-6:
                    chosen_groups_idx = random.sample(range(total_groups), N_groups)
                    chosen_groups_idx.sort() # Keep original chronological order
                else:
                    chosen_groups_idx = list(range(N_groups))

                # Identify which nodes belong to the chosen visible groups
                visible_groups_orig = [current_groups[i] for i in chosen_groups_idx]
                visible_nodes = [node for group in visible_groups_orig for node in group]

                # Filter the time series
                self.time_series = full_time_series[:, visible_nodes]

                # Update the node parent dict to bypass hidden variables 
                full_node_parents = get_parents_dict(global_causal_process)
                self.node_parents_dict = _extract_subgraph(full_node_parents, visible_nodes)

                # Remap group indices to match the new contiguous node array (0 to N_vars)
                self._groups = []
                current_node_idx = 0
                for g in visible_groups_orig:
                    size = len(g)
                    self._groups.append(list(range(current_node_idx, current_node_idx + size)))
                    current_node_idx += size

                # Finally, calculate the macro-parents from the filtered micro-parents
                self.parents_dict = self.extract_group_parents(self.node_parents_dict)
                break
                
            except Exception as e:
                print(f'Generation attempt {it}/{maximum_tries} failed: {str(e)}')
                if it == maximum_tries:
                    raise ValueError(f'Could not generate a dataset after {maximum_tries} tries. Last error: {str(e)}')
            
        if datasets_folder is not None:
            if not os.path.exists(datasets_folder):
                os.makedirs(datasets_folder)
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
    
    def _generate_random_group_links(self, N_groups, density, max_lag, contemp_fraction):
        """Generates a random Macro-Graph if the user does not provide one."""
        group_links = {i: [] for i in range(N_groups)}
        for i in range(N_groups):
            for j in range(N_groups):
                if i == j: continue
                if random.random() < density:
                    # Avoid lag 0 cycles
                    if j > i and random.random() < contemp_fraction:
                        lag = 0
                    else:
                        lag = -random.randint(1, max_lag) if max_lag > 0 else 0
                        if lag == 0: continue # Prevent contemporaneous cycles
                    
                    if (i, lag) not in group_links[j]:
                        group_links[j].append((i, lag))
        return group_links
    
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
    random.seed(0)
    dataset = CausalDataset()
    time_series, parents_dict, groups, node_parents_dict = dataset.generate_group_toy_data(name='test', T=1000, N_vars=50, N_groups=10,
                                inner_group_crosslinks_density=0.2, outer_group_crosslinks_density=0.3,
                                n_node_links_per_group_link=3, contemp_fraction=.1,
                                cross_terms_fraction=0.2,
                                max_lag=3, min_lag=0, dependency_funcs=['linear'],
                                multivariate_funcs=[lambda x, y: x * y],
                                dependency_coeffs=[-0.3, 0.3], auto_coeffs=[0.3, 0.5],
                                noise_dists=['gaussian'], noise_sigmas=[0.2, 1],
                                datasets_folder = None, maximum_tries=100)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for i in range(time_series.shape[1]):
        plt.plot(time_series[:, i] + i*50, label=f'Node {i}')  # Desplazamos cada nodo para visualización
    plt.title('Series Temporales Sintéticas con Estructura Causal de Grupos')
    plt.xlabel('Time')
    plt.ylabel('Value (offset for visibility)')
    plt.legend()
    plt.show()
    
    print('Parents dict (group-level):')
    for group, parents in parents_dict.items():
        print(f'Group {group}: {parents}')
    