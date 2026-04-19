'''
Module with the base class for causal discovery algorithms.
'''

from group_causation.causal_discovery_base import CausalDiscovery

class MicroCausalDiscovery(CausalDiscovery):
    '''
    Class for micro causal discovery algorithms
    Inherits from CausalDiscovery
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        standarize : bool indicating if the data should be standarized. default=True
    '''