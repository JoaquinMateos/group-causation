from typing import Union

import numpy as np
from abc import abstractmethod

from group_causation.causal_discovery_base import CausalDiscovery


class GroupCausalDiscovery(CausalDiscovery): # Abstract class
    '''
    Base class for causal discovery on groups of variables algorithms
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        groups : list[set[int]] list with the sets that will compound each group of variables.
                    We will suppose that the groups are known beforehand.
                    The index of a group will be considered as its position in groups list.
                    By default, each variable is considered a group.
        standarize : bool indicating if the data should be standarized before applying the algorithm.
    '''
    def __init__(self, data: np.ndarray, groups: Union[list[set[int]], None] = None,
                 standarize: bool=True, **kwargs):
        if standarize:
            self._data = data - data.mean(axis=0)
            if np.all((std:=data.std(axis=0))!=0): data /=std
        else:
            self._data = data
        if groups is None:
            self._groups = [[i] for i in range(data.shape[1])]
        else:
            self._groups = [list(group) for group in groups]
        self.extra_args = kwargs

    @abstractmethod
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        To be implemented by subclasses
        
        Returns
            Dictionary with the parents of each group of variables.
        '''
        pass