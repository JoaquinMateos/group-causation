
from typing import Union
import numpy as np
import tigramite
import tigramite.data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

# To admit the use of this package's data structures
from causalai.data.time_series import TimeSeriesData

from group_causation.micro_causal_discovery.micro_causal_discovery_base import MicroCausalDiscovery

class PCMCIWrapper(MicroCausalDiscovery):
    '''
    Wrapper for PCMCI algorithm
    
    Args:
        data: np.array with the data, shape (n_samples, n_features)
        cond_ind_test: string with the name of the conditional independence test
        min_lag: minimum lag to consider
        max_lag: maximum lag to consider
        pc_alpha: alpha value for the conditional independence test
    '''
    def __init__(self, data: np.ndarray, cond_ind_test='parcorr',
                 min_lag=1, max_lag=3, pc_alpha: int = None, **kwargs):
        '''
        Initialize the PCMCI object
        
        '''
        super().__init__(data, **kwargs)
        
        self.cond_ind_test = {'parcorr': ParCorr(),
                              'gpdc': GPDC(),
                              'cmiknn': CMIknn(significance='fixed_thres'), # Very slow
                              }[cond_ind_test]
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.pc_alpha = pc_alpha
        self.extra_args = kwargs
        
        dataframe = convert_to_tigramite_dataframe(self._data)
        self.pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=self.cond_ind_test,
            verbosity=0,
        )
    
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        Returns the parents dict
        
        :param data: np.array with the data, shape (n_samples, n_features)
        '''
        results = self.pcmci.run_pcmciplus(tau_min=self.min_lag, tau_max=self.max_lag,
                                            pc_alpha=self.pc_alpha, **self.extra_args)
        parents = self.pcmci.return_parents_dict(graph=results['graph'], 
                                                 val_matrix=results['val_matrix'],
                                                 include_lagzero_parents=True)
        return parents



class LPCMCIWrapper(MicroCausalDiscovery):
    '''
    Wrapper for LPCMCI algorithm
    
    Args:
        data: np.array with the data, shape (n_samples, n_features)
        cond_ind_test: string with the name of the conditional independence test
        min_lag: minimum lag to consider
        max_lag: maximum lag to consider
        pc_alpha: alpha value for the conditional independence test
    '''
    def __init__(self, data, cond_ind_test='parcorr',
                 min_lag=1, max_lag=3, pc_alpha=0.05, **kwargs):
        '''
        Initialize the PCMCI object
        
        '''
        super().__init__(data, **kwargs)
        
        self.cond_ind_test = {'parcorr': ParCorr(significance='analytic'),
                              }[cond_ind_test]
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.pc_alpha = pc_alpha
        
        dataframe = convert_to_tigramite_dataframe(self._data)
        self.lpcmci = LPCMCI(
            dataframe=dataframe,
            cond_ind_test=self.cond_ind_test,
            verbosity=0
        )
    
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        Returns the parents dict
        '''
        self.lpcmci.run_lpcmci(tau_min=self.min_lag, tau_max=self.max_lag, pc_alpha=self.pc_alpha)
        
        # Return parents dict manually
        parents = dict()
        for j in range(self.lpcmci.N):
            for ((i, lag_i), link) in self.lpcmci.graph_dict[j].items():
                if len(link) > 0 and (lag_i < 0 or i < j):
                    parents[j] = parents.get(j, []) + [(i, lag_i)]
        
        return parents
    
class PCStableWrapper(MicroCausalDiscovery):
    '''
    Wrapper for PC Stable algorithm
    
    Args:
        data: np.array with the data, shape (n_samples, n_features)
        cond_ind_test: string with the name of the conditional independence test
        min_lag: minimum lag to consider
        max_lag: maximum lag to consider
        pc_alpha: alpha value for the conditional independence test
    '''
    def __init__(self, data: np.ndarray, cond_ind_test='parcorr', 
                 min_lag=1, max_lag=3, pc_alpha=0.05, **kwargs):
        super().__init__(data, **kwargs)
        
        self.cond_ind_test = {'parcorr': ParCorr(significance='analytic'),
                              }[cond_ind_test]
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.pc_alpha = pc_alpha
        
        dataframe = convert_to_tigramite_dataframe(self._data)
        self.pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=self.cond_ind_test,
            verbosity=0
        )

    def extract_parents(self) -> dict[int, list[int]]:
        '''
        Returns the parents dict
        '''
        parents = self.pcmci.run_pc_stable(tau_min=self.min_lag, tau_max=self.max_lag, pc_alpha=self.pc_alpha)
        
        
        return parents

def convert_to_tigramite_dataframe(data: Union[TimeSeriesData, np.ndarray]) -> tigramite.data_processing.DataFrame:
    '''
    Convert the data to tigramite dataframe format
    Note: It only works if there is only one data array in the data object
    '''
    if isinstance(data, TimeSeriesData):
        names = data.var_names
        data_arrays = data.data_arrays
        dataframe = tigramite.data_processing.DataFrame(data_arrays[0], var_names=names)
    
    elif isinstance(data, np.ndarray):
        dataframe = tigramite.data_processing.DataFrame(data, var_names=[str(i) for i in range(data.shape[1])])
    
    return dataframe
