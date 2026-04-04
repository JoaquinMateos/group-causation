import numpy as np
from sklearn.decomposition import PCA
from typing import Any, Union

from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.micro_causal_discovery.causal_discovery_causalnex import DynotearsWrapper
from group_causation.micro_causal_discovery.causal_discovery_tigramite import PCMCIWrapper, PCStableWrapper
from group_causation.micro_causal_discovery.micro_causal_discovery_base import MicroCausalDiscovery

class DimensionReductionGroupCausalDiscovery(GroupCausalDiscovery):
    '''
    Class that implements the dimension reduction algorithm for causal discovery on groups of variables.
    
    The constructor prepares the groups of variables using a dimensionality reduction technique,
    and then applies a causal discovery algorithm to discover the causal relationships between the variables of each group.
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        groups : list[set[int]] list with the sets that will compound each group of variables.
                    We will suppose that the groups are known beforehand.
                    The index of a group will be considered as its position in groups list.
        dimensionality_reduction : str indicating the type of dimensionality reduction technique
                    that is applied to groups. options=['pca']. default='pca'
        node_causal_discovery_alg : str indicating the algorithm that will be used to discover the causal
                    relationships between the variables of each group. options=['pcmci', 'pc-stable', 'dynotears']
    '''
    def __init__(self, data: np.ndarray,
                    groups: list[set[int]],
                    dimensionality_reduction: str = 'pca',
                    node_causal_discovery_alg: str = 'pcmci',
                    node_causal_discovery_params: Union[dict[Any, Any], None] = None,
                    **kwargs):
        super().__init__(data, groups, **kwargs)
        
        self.node_causal_discovery_alg = node_causal_discovery_alg
        self.node_causal_discovery_params = node_causal_discovery_params if node_causal_discovery_params is not None else {}
        self.extra_args = kwargs
        
        self._groups_data = self._prepare_groups_data(dimensionality_reduction)
    
    def _prepare_groups_data(self, dimensionality_reduction: str) -> np.ndarray:
        '''
        Execute the indicate dimensionality reduction algorithm to the groups of variables,
        in order to obtain a univariate time series for each group.
        
        Args:
            dimensionality_reduction : str indicating the type of dimensionality reduction technique
                        that is applied to groups. options=['pca']. default='pca'
        
        Returns:
            groups_data : np.ndarray where each column is the univariate time series of each group
                            of variables after the dimensionality reduction
        '''
        groups_data = []
        for group in self._groups:
            group_data = self._data[:, list(group)]
            if dimensionality_reduction == 'pca':
                pca = PCA(n_components=1)
                group_data = pca.fit_transform(group_data)
            elif dimensionality_reduction == 'avg':
                group_data = np.mean(group_data, axis=0)
            else:
                raise ValueError(f'Invalid dimensionality reduction technique: {dimensionality_reduction}')
            groups_data.append(group_data)
        
        time_series = np.array(groups_data).reshape(len(groups_data), -1).T
        time_series = np.concatenate(groups_data, axis=1)
        return time_series
    
    def extract_parents(self) -> dict[int, list[Union[int, tuple[int, int]]]]:
        '''
        Extract the parents of each group of variables using the dimension reduction algorithm
        
        Returns
            Dictionary with the parents of each group of variables.
        '''
        self.causal_discovery_alg = self._getCausalDiscoveryAlgorithm()
        
        group_parents = self.causal_discovery_alg.extract_parents()
                
        return group_parents

    def _getCausalDiscoveryAlgorithm(self) -> MicroCausalDiscovery:
        '''
        Get the causal discovery algorithm that will be used to discover the causal relationships
        between the variables of each group.
        
        Returns:
            causal_discovery_alg : function that will be used to discover the causal relationships
        '''
        if self.node_causal_discovery_alg == 'pcmci':
            return PCMCIWrapper(data=self._groups_data, **self.node_causal_discovery_params)
        elif self.node_causal_discovery_alg == 'pc-stable':
            return PCStableWrapper(data=self._groups_data, **self.node_causal_discovery_params)
        elif self.node_causal_discovery_alg == 'dynotears':
            return DynotearsWrapper(data=self._groups_data, **self.node_causal_discovery_params)
        else:
            raise ValueError(f'Invalid node causal discovery algorithm: {self.node_causal_discovery_alg}')