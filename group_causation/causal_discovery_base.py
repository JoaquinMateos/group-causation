'''
Module with the base class for causal discovery algorithms.
'''


import os
import threading
import time
import numpy as np
from abc import ABC, abstractmethod
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
            parents = self.extract_parents()
        except Exception as e:
            # Stop the memory monitor in case of error
            keep_measuring = False
            monitor_thread.join()
            toc = time.time()
            
            logging.error(f"Error executing {self.__class__.__name__}: {str(e)}")
            return {}, (toc - tic), -1.0

        # Stop the memory monitor after the algorithm finishes
        keep_measuring = False
        monitor_thread.join()
        
        toc = time.time()
        execution_time = toc - tic
        
        # Calculate the difference between the maximum peak and the initial memory
        memory_used_bytes = peak_memory[0] - mem_base
        memory_used_mb = memory_used_bytes / (1024 * 1024) # Convert to MiB
        
        return parents, execution_time, memory_used_mb

