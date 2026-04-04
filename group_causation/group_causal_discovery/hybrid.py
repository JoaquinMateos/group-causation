import numpy as np
from sklearn.decomposition import PCA
from typing import Any, Union
import logging
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.group_causal_discovery.micro_level import MicroLevelGroupCausalDiscovery

class HybridGroupCausalDiscovery(GroupCausalDiscovery):
    '''
    Class that implements a group causal discovery algorithm which combines dimension reduction
    techniques with microlevel causal discovery.
    To do so, given a set of groups of variables, the algorithm will apply a dimension reduction
    technique to reduce the dimensionality of the problem. However, the dimension reduction won't
    necessarily give a one-dimensional time series, but it might give a set of time series (microgroups)
    with less variables than the original problem. Then, a microlevel causal discovery algorithm will
    be applied to the reduced time series, and the group causal graph will be extracted from the
    microgroups one.
    
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        groups : list[set[int]] list with the sets that will compound each group of variables.
                We will suppose that the groups are known beforehand.
                The index of a group will be considered as its position in groups list.
        dimensionality_reduction : str indicating the type of dimensionality reduction technique
                that is applied to groups. options=['pca']. default='pca'
        dimensionality_reduction_params : dict with the parameters for the dimensionality reduction algorithm.
        node_causal_discovery_alg : str indicating the algorithm that will be used to discover the causal
                relationships between the variables of each group. options=['pcmci', 'pc-stable', 'dynotears']
        node_causal_discovery_params : dict with the parameters for the node causal discovery algorithm.
        link_assumptions (dict) : Dictionary of form {j:{(i, -tau): link_type, …}, …} specifying assumptions about links.
                This initializes the graph with entries graph[i,j,tau] = link_type. For example, graph[i,j,0] = ‘–>’ 
                implies that a directed link from i to j at lag 0 must exist. Valid link types are ‘o-o’, ‘–>’, ‘<–’.
                In addition, the middle mark can be ‘?’ instead of ‘-’. Then ‘-?>’ implies that this link may not 
                exist, but if it exists, its orientation is ‘–>’. Link assumptions need to be consistent, i.e., 
                graph[i,j,0] = ‘–>’ requires graph[j,i,0] = ‘<–’ and acyclicity must hold. If a link does not appear
                in the dictionary, it is assumed absent. That is, if link_assumptions is not None, then all links have 
                to be specified or the links are assumed absent.
    '''
    def __init__(self, data: np.ndarray,
                    groups: list[set[int]],
                    dimensionality_reduction_params: dict[str, Any],
                    link_assumptions: Union[dict[int, dict[tuple[int, int], str]], None] = None,
                    dimensionality_reduction: str = 'pca',
                    node_causal_discovery_alg: str = 'pcmci',
                    node_causal_discovery_params: Union[dict[str, Any], None] = None,
                    verbose: int = 0,
                    **kwargs):
        super().__init__(data, groups, **kwargs)
        
        self._node_causal_discovery_alg = node_causal_discovery_alg
        self._node_causal_discovery_params = node_causal_discovery_params if node_causal_discovery_params is not None else {}
        self._verbose = verbose
        
        if dimensionality_reduction == 'pca':
            # Normalize data before applying PCA
            self._data = (self._data - self._data.mean(axis=0))
            if np.all((std:=self._data.std(axis=0))!=0): self._data /= std
            self.micro_groups, self.micro_data = self._prepare_micro_groups_pca(**dimensionality_reduction_params)
        else:
            raise ValueError(f'Dimensionality reduction technique {dimensionality_reduction} not supported.')
        
        micro_link_assumptions = _convert_link_assumptions(link_assumptions, self.micro_groups)
        
        self._node_causal_discovery_params['link_assumptions'] = micro_link_assumptions
        self.micro_level_causal_discovery = MicroLevelGroupCausalDiscovery(self.micro_data, self.micro_groups,
                                                                    self._node_causal_discovery_alg, self._node_causal_discovery_params)
    
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        '''
        Extract the parents of each group of variables using the dimension reduction algorithm
        
        Returns:
            Dictionary with the parents of each group of variables.
        '''
        group_parents = self.micro_level_causal_discovery.extract_parents()
        
        if self._verbose > 1:
            logging.info(f'Parents extracted: {group_parents}')
        
        return group_parents
    
    
    def _prepare_micro_groups_pca(self, explained_variance_threshold: float = 0.5,
                                    embedding_ratio: Union[float, None] = None,
                                    embedding_size: Union[int, None] = None,
                                    groups_division_method: str='group_embedding') -> tuple[list[set[int]], np.ndarray]:
        '''
        Execute the PCA dimensionality reduction algorithm to the groups of variables,
        in order to obtain a univariate time series for each group.
        
        Args:
            explained_variance_threshold : float indicating the minimum explained variance that the PCA
                        algorithm must achieve to stop the dimensionality reduction.
            embedding_ratio : float indicating the ratio between the number of variables in the original dataset
                        and the number of variables in the reduced dataset. If None, the explained_variance_threshold
                        will be used to calculate the explained variance threshold.
            groups_compresion_method : string indicating the method that will be used to compress the
                        groups of variables. options=['group_embedding', 'subgroups']
        
        Returns:
            micro_groups : list[ set[int] ] where keys are original groups, and values are the indexes of
                                    associated microvariables.
            micro_groups_data : np.ndarray where each column is the univariate time series of each group
                            of variables after the dimensionality reduction
        '''
        if embedding_ratio is not None and embedding_size is not None:
            raise ValueError('Only one of embedding_ratio or embedding_size can be specified.')
        if embedding_ratio is not None:
            self._explained_variance_threshold = self._get_variance_threshold_from_embedding_ratio_pca(embedding_ratio)
        elif embedding_size is not None:
            self._explained_variance_threshold = self._get_variance_threshold_from_embedding_size_pca(embedding_size)
        else:
            self._explained_variance_threshold = explained_variance_threshold
        # Admit a low error when explained_variance_threshold is 0.0
        if self._explained_variance_threshold == 0.0:
            self._explained_variance_threshold = 0.05

        if self._explained_variance_threshold < 0 or self._explained_variance_threshold >= 1:
            raise ValueError(f'Explained variance threshold must be between 0 and 1. Obtained: {self._explained_variance_threshold}.\n'
                             'Note that if you specified embedding_ratio, the explained variance threshold will be calculated from it.')
        else:
            self._explained_variance_threshold = float(self._explained_variance_threshold)
        
        micro_groups = []
        micro_data = [] # List where each element is the ts data of a microgroup
        for group in self._groups:
            
            if groups_division_method == 'group_embedding':
                current_number_of_variables = sum(arr.shape[1] for arr in micro_data)
                
                micro_group, group_data_pca = self._get_group_embedding(group, current_number_of_variables)
                micro_groups.append(micro_group)
                micro_data.append(group_data_pca)
                
            elif groups_division_method == 'subgroups':                
                micro_group, group_data_pca = self._divide_subgroups(group)
                micro_groups.append( set(micro_group) )
                micro_data.append(group_data_pca)
            
            else:
                raise ValueError(f'Invalid groups division method: {groups_division_method}')
        
        micro_data = np.concatenate(micro_data, axis=1)
        
        if self._verbose > 0:
            print(f'Data dimensionality has been reduced to {micro_data.shape[1]} in order to perform microlevel causal discovery.')

        return micro_groups, micro_data

    def _get_group_embedding(self, group: list[int], current_number_of_variables: int) -> tuple[set[int], np.ndarray]:
        '''
        Function that applies PCA to a group of variables and returns the microgroup
        and the data of the microgroup.
        
        Args:
            group : set[int] indicating the indexes of the variables in the group.
            current_number_of_variables : int indicating the current number of variables in the microlevel data.
        
        Returns:
            micro_group : set[int] indicating the indexes of the variables in the microgroup.
            group_data_pca : np.ndarray where each column is the univariate time series of each microvariable
                            of the group after the dimensionality reduction
        '''
        group_data = self._data[:, list(group)]
                
        # Extract the principal components of the group
        pca = PCA(n_components=self._explained_variance_threshold)
        group_data_pca = pca.fit_transform(group_data)

        # Create the microgroup
        n_variables = group_data_pca.shape[1]
        micro_group = set(range(current_number_of_variables,
                                current_number_of_variables + n_variables))
        
        return micro_group, group_data_pca
    
    def _divide_subgroups(self, current_subgroup: list[int], current_number_of_variables: int = 0) -> tuple[list[int], np.ndarray]:
        '''
        Recursive function that divides the group in 2 subgroups until the explained variance
        of the first PC represents at least a "explained_variance_threshold" fraction of the total
        '''
        current_subgroup_data = self._data[:, list(current_subgroup)]
        pca = PCA(n_components=1)
        group_data_pca = pca.fit_transform(current_subgroup_data)
        if pca.explained_variance_ratio_[0] >= self._explained_variance_threshold or len(current_subgroup) == 1:
            # We have reached the desired explained variance; one single pc is enough
            used_subgroup = [current_number_of_variables]
            current_number_of_variables += 1
            return used_subgroup, group_data_pca
        else:
            # Divide the half of the variables that have highest importance in PC1
            ordered_nodes = np.argsort(pca.components_[0])
            half = len(current_subgroup) // 2
            first_half = ordered_nodes[:half]
            second_half = ordered_nodes[half:]
            first_subgroup, first_subgroup_data = self._divide_subgroups(
                [current_subgroup[i] for i in first_half],
                current_number_of_variables,
            )
            second_subgroup, second_subgroup_data = self._divide_subgroups(
                [current_subgroup[i] for i in second_half],
                current_number_of_variables + len(first_subgroup),
            )
            return first_subgroup + second_subgroup, np.concatenate([first_subgroup_data, second_subgroup_data], axis=1)
    
    def _get_variance_threshold_from_embedding_ratio_pca(self, embedding_ratio: Union[float, None] = None) -> float:
        '''
        Function that calculates the explained variance threshold from the embedding ratio.
        The embedding ratio is the ratio between the number of variables in the original dataset
        and the number of variables in the reduced dataset.
        '''
        if embedding_ratio is None:
            raise ValueError('embedding_ratio must be provided when using embedding_ratio mode.')

        variance_thresholds = []
        for group in self._groups:
            group_data = self._data[:, list(group)]
            pca = PCA(n_components=int( embedding_ratio * len(group) ))
            pca.fit(group_data)
            explained_variance = pca.explained_variance_ratio_.sum()
            variance_thresholds.append(explained_variance)
        explained_variance_threshold = float(np.mean(variance_thresholds))
        
        return explained_variance_threshold
    
    
    def _get_variance_threshold_from_embedding_size_pca(self, embedding_size: Union[int, None] = None) -> float:
        '''
        Function that calculates the explained variance threshold from the embedding size.
        The embedding ratio is the ratio between the number of variables in the original dataset
        and the number of variables in the reduced dataset.
        '''
        if embedding_size is None:
            raise ValueError('embedding_size must be provided when using embedding_size mode.')

        variance_thresholds = []
        for group in self._groups:
            if embedding_size >= len(group):
                variance_thresholds.append(1)
                continue
            group_data = self._data[:, list(group)]
            pca = PCA(n_components=int( embedding_size ))
            pca.fit(group_data)
            explained_variance = pca.explained_variance_ratio_.sum()
            variance_thresholds.append(explained_variance)
        explained_variance_threshold = float(np.mean(variance_thresholds))
        
        return explained_variance_threshold
    



def _convert_link_assumptions(link_assumptions: Union[dict[int, dict[tuple[int, int], str]], None], micro_groups: list[set[int]]) -> Union[dict[int, dict[tuple[int, int], str]], None]:
    '''
    Convert the link assumptions from the original groups to the microgroups
    
    Args:
        link_assumptions : dict[int, dict[tuple[int, int], str]]. Dictionary with the link assumptions.
        micro_groups : list[ set[int] ]. List with the microgroups.
    
    Returns:
        micro_link_assumptions : dict[int, dict[tuple[int, int], str]]. Dictionary with the link assumptions for each microgroup.
    '''
    if link_assumptions is None:
        return None
    
    micro_link_assumptions = {}
    for son_group_idx, son_group in enumerate(micro_groups):
        for son_node_idx in son_group:
            if son_node_idx not in micro_link_assumptions:
                micro_link_assumptions[son_node_idx] = {}
            for (parent_group_idx, lag), link_type in link_assumptions[son_group_idx].items():
                for parent_node_idx in micro_groups[parent_group_idx]:
                    micro_link_assumptions[son_node_idx][(parent_node_idx, lag)] = link_type
    
    return micro_link_assumptions