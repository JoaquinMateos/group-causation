'''
Module with the base class for causal discovery algorithms.
'''


import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union, cast
from memory_profiler import memory_usage
import logging
import psutil

class CausalDiscovery(ABC): # Abstract class
    '''
    Base class for causal discovery algorithms
    
    To be implemented by subclasses
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        standarize : bool indicating if the data should be standarized. default=True
    '''
    @abstractmethod
    def __init__(self, data: np.ndarray, standarize: bool=True, **kwargs):
        if standarize:
            self._data = data - data.mean(axis=0)
            if np.all((std:=data.std(axis=0))!=0): data /=std
        else:
            self._data = data
    
    @abstractmethod
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        '''
        To be implemented by subclasses
        
        Returns
            Dictionary with the parents of each node
        '''
        pass
    
    def extract_parents_time_and_memory(self) -> tuple[dict[int, list[tuple[int, int]]], float, float]:
        '''
        Execute the extract_parents method and return the parents dict, the time that took to run the algorithm
        
        Returns:
            parents : dictionary of extracted parents
            execution_time : execution time in seconds
            memory : volatile memory used by the process, in MB
        '''
        tic = time.time()
        logging.info(f"Running {self.__class__.__name__}...")
        try:
            memory, parents = memory_usage(
                cast(Any, self.extract_parents),
                retval=True,
                include_children=True,
                multiprocess=True,
            )
        except psutil.NoSuchProcess:
            toc = time.time()
            execution_time = toc - tic
            
            error_message = (
                f"[OOM KILLED] The Operative System killed {self.__class__.__name__}. "
                f"The system ran out of memory. Try reducing the 'batch_size' or the 'max_lag'."
            )
            print(error_message, flush=True)
            # Return empty parents dict, execution time until OOM, and -1 for memory to indicate OOM
            return {}, execution_time, -1.0 
            
        except Exception as e:
            # Capture any other unexpected error so it doesn't break the loop of datasets
            toc = time.time()
            execution_time = toc - tic
            error_message = (
                f"[UNEXPECTED ERROR] Error running {self.__class__.__name__}: {str(e)}"
            )
            print(error_message, flush=True)
            return {}, execution_time, -1.0

        toc = time.time()
        execution_time = toc - tic
        
        memory_used = max(memory) - min(memory)  # Memory usage in MiB
        memory_used = memory_used * 1.048576 # Exact division  1024^2 / 1000^2
        
        return parents, execution_time, memory_used

