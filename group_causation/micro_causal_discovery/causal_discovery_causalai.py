from typing import Any
import numpy as np
from causalai.models.time_series.granger import Granger
from causalai.models.time_series.var_lingam import VARLINGAM

# To admit the use of this package's data structures
from causalai.data.time_series import TimeSeriesData

from group_causation.micro_causal_discovery import MicroCausalDiscovery

class GrangerWrapper(MicroCausalDiscovery):
    '''
    Wrapper for Granger algorithm
    
    Args:
        data : np.array with the data, shape (n_samples, n_features)
        min_lag : minimum lag to consider
        max_lag : maximum lag to consider
        cv : number of folds for the cross-validation
    '''
    def __init__(self, data: np.ndarray, min_lag=1,
                 max_lag=3, cv=5, **kwargs):
        super().__init__(data, **kwargs)
        
        self.min_lag = min_lag
        self.max_lag = max_lag
        
        self.granger = Granger(TimeSeriesData(self._data), cv=cv, max_iter=1e5)

    def extract_parents(self) -> dict[int, list[int]]:
        '''
        Returns the parents dict
        Args:
            data : np.array with the data, shape (n_samples, n_features)
        '''
        parents_causalai = self.granger.run(max_lag=self.max_lag)
        parents_dict = get_parents_dict(parents_causalai)
        
        return parents_dict

class VARLINGAMWrapper(MicroCausalDiscovery):
    '''
    Wrapper for VARLINGAM algorithm
    '''
    def __init__(self, data: np.ndarray, min_lag=1,
                 max_lag=3, **kwargs):
        '''
        Initialize the VARLINGAM object
        Args:
            data : np.array with the data, shape (n_samples, n_features)
            min_lag : minimum lag to consider
            max_lag : maximum lag to consider
        '''
        super().__init__(data, **kwargs)
        
        self.min_lag = min_lag
        self.max_lag = max_lag
        
        self.varlingam = VARLINGAM(TimeSeriesData(self._data))
    
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        Returns the parents dict
        
        Args:
            data : np.array with the data, shape (n_samples, n_features)
        '''
        parents_causalai = self.varlingam.run(max_lag=self.max_lag)
        parents_dict = get_parents_dict(parents_causalai)
        
        return parents_dict


def get_parents_dict(parents_causalai: dict[str, dict[str, Any]]):
    '''
    Convert the parents dict from CausalAI format to the format used in the benchmarks
    '''
    parents_dict = {node: values['parents'] for node, values in parents_causalai.items()}
    
    return parents_dict