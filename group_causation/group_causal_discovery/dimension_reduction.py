import numpy as np
from sklearn.decomposition import PCA
from typing import Any

from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.micro_causal_discovery.causal_discovery_causalnex import DynotearsWrapper
from group_causation.micro_causal_discovery.causal_discovery_tigramite import PCMCIWrapper
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
                    pca_n_components: int = 1,
                    node_causal_discovery_alg: str = 'pcmci',
                    node_causal_discovery_params: dict[Any, Any] | None = None,
                    **kwargs):
        super().__init__(data, groups, **kwargs)
        
        self.node_causal_discovery_alg = node_causal_discovery_alg
        self.pca_n_components = pca_n_components
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
                pca = PCA(n_components=min(self.pca_n_components, len(group)))
                group_data = pca.fit_transform(group_data)
            elif dimensionality_reduction == 'avg':
                group_data = np.mean(group_data, axis=1, keepdims=True)
            else:
                raise ValueError(f'Invalid dimensionality reduction technique: {dimensionality_reduction}')
            groups_data.append(group_data)
        
        self._group_embedding_sizes = [embedding.shape[1] for embedding in groups_data]
        time_series = np.concatenate(groups_data, axis=1)
        return time_series

    def get_recovered_latents(self) -> list[np.ndarray]:
        """Return the per-group reduced embedding as numpy arrays (for MCC evaluation)."""
        latents = []
        start = 0
        for size in self._group_embedding_sizes:
            latents.append(self._groups_data[:, start:start + size])
            start += size
        return latents
    
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        '''
        Extract the parents of each group of variables using the dimension reduction algorithm
        
        Returns
            Dictionary with the parents of each group of variables.
        '''
        self.causal_discovery_alg = self._getCausalDiscoveryAlgorithm()
        
        component_parents = self.causal_discovery_alg.extract_parents()
                
        return self._convert_component_to_group_parents(component_parents)

    def _convert_component_to_group_parents(
            self, component_parents: dict[int, list[tuple[int, int]]],
    ) -> dict[int, list[tuple[int, int]]]:
        """Collapse the graph over components into the graph over groups.

        Each group owns a contiguous block of components (see
        ``_group_embedding_sizes``), so several components of the same group
        become a single group-level node.
        """
        component_to_group: dict[int, int] = {}
        for group_index, size in enumerate(self._group_embedding_sizes):
            for component in range(len(component_to_group), len(component_to_group) + size):
                component_to_group[component] = group_index

        group_parents: dict[int, list[tuple[int, int]]] = {group: [] for group in range(len(self._groups))}
        for child, parents in component_parents.items():
            child_group = component_to_group[child]
            for parent in parents:
                parent_node = parent[0] if isinstance(parent, tuple) else parent
                parent_lag = parent[1] if isinstance(parent, tuple) else 0
                edge = (component_to_group[parent_node], parent_lag)
                if edge == (child_group, 0):
                    continue
                if edge not in group_parents[child_group]:
                    group_parents[child_group].append(edge)
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
        elif self.node_causal_discovery_alg == 'dynotears':
            return DynotearsWrapper(data=self._groups_data, **self.node_causal_discovery_params)
        else:
            raise ValueError(f'Invalid node causal discovery algorithm: {self.node_causal_discovery_alg}')