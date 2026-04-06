import logging

import numpy as np
from typing import Any, Union

from group_causation.micro_causal_discovery.micro_causal_discovery_base import MicroCausalDiscovery
from group_causation.micro_causal_discovery.causal_discovery_causalnex import DynotearsWrapper
from group_causation.micro_causal_discovery.causal_discovery_tigramite import PCMCIWrapper, PCStableWrapper
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery


class MicroLevelGroupCausalDiscovery(GroupCausalDiscovery):
    '''
    Class that implements the dimension reduction algorithm for causal discovery on groups of variables.
    A causal discovery algorithm is applied to discover the causal relationships between all of the node-level
    variables, and then the whole graph is reduced to the group-level graph.
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        groups : list[set[int]] list with the sets that will compound each group of variables.
                    We will suppose that the groups are known beforehand.
                    The index of a group will be considered as its position in groups list.
        node_causal_discovery_alg : str indicating the algorithm that will be used to discover the causal
                    relationships between the variables of each group. options=['pcmci', 'pc-stable', 'dynotears']
        node_causal_discovery_params : dict with the parameters for the node causal discovery algorithm.
    '''
    def __init__(self, data: np.ndarray,
                    groups: list[set[int]],
                    node_causal_discovery_alg: str = 'pcmci',
                    node_causal_discovery_params: Union[dict[str, Any], None] = None,
                    **kwargs):
        super().__init__(data, groups, **kwargs)
        
        self.node_causal_discovery_alg = node_causal_discovery_alg
        self.node_causal_discovery_params = node_causal_discovery_params if node_causal_discovery_params is not None else {}
        
    
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        '''
        Extract the parents of each group of variables using the micro-level causal discovery algorithm
        to extract the micro-level causal DAG and convert it in a group-level graph.
        
        Returns
            Dictionary with the parents of each group of variables.
        '''
        self.causal_discovery_alg = self._getCausalDiscoveryAlgorithm()
        
        node_parents = self.causal_discovery_alg.extract_parents()

        group_parents = self._convert_node_to_group_parents(node_parents)
        return group_parents

    def _convert_node_to_group_parents(self, node_parents: dict[int, list[tuple[int, int]]]) -> dict[int, list[tuple[int, int]]]:
        '''
        Convert the parents of each node to the parents of each group of variables
        
        Args:
            node_parents : dict[int, list[tuple[int, int]]]. Dictionary with the parents of each node.
        
        Returns:
            group_parents : dict[int, list[tuple[int, int]]]. Dictionary with the parents of each group of variables.
        '''
        group_parents = {}

        # Build an index from node to all groups containing that node
        node_to_groups = {}
        for idx, group in enumerate(self._groups):
            for node_idx in group:
                node_to_groups.setdefault(node_idx, set()).add(idx)

        for group_idx, group in enumerate(self._groups):
            parent_groups = set()
            for son_node_idx in group:
                # A group is son of another group iff any node has a parent node that is in the parent group
                for parent_node in node_parents.get(son_node_idx, []):
                    parent_node_idx = parent_node[0] if isinstance(parent_node, tuple) else parent_node
                    parent_node_offset = parent_node[1] if isinstance(parent_node, tuple) else 0
                    # Add all matching parent groups (supports overlapping groups safely)
                    for parent_group_idx in node_to_groups.get(parent_node_idx, set()):
                        parent_groups.add((parent_group_idx, parent_node_offset))
            # Remove duplicates
            group_parents[group_idx] = list(parent_groups)
        
        return group_parents
    
    def _getCausalDiscoveryAlgorithm(self) -> MicroCausalDiscovery:
        '''
        Get the causal discovery algorithm that will be used to discover the causal relationships
        between the variables of each group.
        
        Returns:
            causal_discovery_alg : function that will be used to discover the causal relationships
        '''
        if self.node_causal_discovery_alg == 'pcmci':
            return PCMCIWrapper(data=self._data, **self.node_causal_discovery_params)
        elif self.node_causal_discovery_alg == 'pc-stable':
            return PCStableWrapper(data=self._data, **self.node_causal_discovery_params)
        elif self.node_causal_discovery_alg == 'dynotears':
            return DynotearsWrapper(data=self._data, **self.node_causal_discovery_params)
        else:
            raise ValueError(f'Invalid node causal discovery algorithm: {self.node_causal_discovery_alg}')