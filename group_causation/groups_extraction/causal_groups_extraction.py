import logging
import os
import threading
import time
import numpy as np
from abc import ABC, abstractmethod
import psutil



class CausalGroupsExtractorBase(ABC): # Abstract class
    '''
    Base class to extract a set of groups of variables that may be used to later
    predict the causal structure over these groups of variables
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        standarize : bool indicating if the data should be standarized before applying the algorithm
    '''
    def __init__(self, data: np.ndarray, standarize: bool=True, **kwargs):
        if standarize:
            self._data = (data - data.mean(axis=0)) / data.std(axis=0)
        else:
            self._data = data
        self.extra_args = kwargs
    
    @abstractmethod
    def extract_groups(self) -> list[set[int]]:
        '''
        To be implemented by subclasses
        
        Returns
            groups : list of sets with the variables that compound each group
        '''
        pass
    
    def extract_groups_time_and_memory(self) -> tuple[list[set[int]], float, float]:
        '''
        Execute the extract_groups method and return the set of groups, the time that took to run the algorithm
        
        Returns:
            groups : list of sets with the variables that compound each group
            execution_time : execution time in seconds
            memory : volatile memory used by the process, in MB
        '''
        process = psutil.Process(os.getpid())
        mem_base = process.memory_info().rss
        peak_memory = [mem_base]
        keep_measuring = True

        # Create a thread to monitor memory usage while the algorithm is running
        def monitor_memory():
            while keep_measuring:
                try:
                    current_mem = process.memory_info().rss
                    if current_mem > peak_memory[0]:
                        peak_memory[0] = current_mem
                    time.sleep(0.05) # Check memory every 50ms
                except psutil.NoSuchProcess:
                    break

        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()

        tic = time.time()
        
        try:
            groups = self.extract_groups()
        except Exception as e:
            # Stop the memory monitor in case of error
            keep_measuring = False
            monitor_thread.join()
            toc = time.time()
            
            logging.error(f"Error executing {self.__class__.__name__}: {str(e)}")
            return [], (toc - tic), -1.0

        # Stop the memory monitor after the algorithm finishes
        keep_measuring = False
        monitor_thread.join()
        
        toc = time.time()
        execution_time = toc - tic
        
        # Calculate the difference between the maximum peak and the initial memory
        memory_used_bytes = peak_memory[0] - mem_base
        
        # Exact division to MB (Megabytes). 
        # Equivalent to the (MiB * 1.048576) from your original code.
        memory_used_mb = memory_used_bytes / 1_000_000 
        
        return groups, execution_time, memory_used_mb