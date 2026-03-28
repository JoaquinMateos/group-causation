'''
Code for implementation of the ADAG (Adaptative Aggregation) algorithm.
Implements the method described in:

    - Urmi Ninad, Jonas Wahl*, Andreas Gerhadus* and Jakob Runge.
    Causaly on vector-valued variables and consistency-guided aggregation. 2025.
'''

from typing import Any
import numpy as np
from group_causation.group_causal_discovery import HybridGroupCausalDiscovery
from group_causation.group_causal_discovery.micro_level import MicroLevelGroupCausalDiscovery
from dowhy.gcm.independence_test import independence_test  # type: ignore[reportMissingImports]

class ADAG(HybridGroupCausalDiscovery):
    '''
    Class that implements the ADAG Group Causal Discovery algorithm.
        
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.node_causal_discovery_alg = kwargs.get('node_causal_discovery_alg', 'pcmci')
        self.node_causal_discovery_params = kwargs.get('node_causal_discovery_params', {})
    
    
    def _get_aggregation_independence_score(self) -> float:
        '''
        Function that computes the independence score of the aggregation.
        It is calculated as |C^ind| / (|C^ind| + |I^ind|), where
            - C^ind is the set of consistent independencies between original variables and their aggregations.
            - I^ind is the set of independencies that are in the aggregated data but not in the original data.
        '''
        raise NotImplementedError('ADAG independence score is not implemented yet.')

