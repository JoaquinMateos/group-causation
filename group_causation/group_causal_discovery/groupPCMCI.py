import numpy as np
import itertools
import logging
import torch
import math
from scipy.stats import chi2
from typing import Any, Union, Optional

from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.independence_tests import conditional_independence_tests
from group_causation.independence_tests.conditional_independence_base import ConditionalIndependence_base

class GroupPCMCICausalDiscovery(GroupCausalDiscovery):
    '''
    Group-level causal discovery algorithm using the PCMCI framework.
    Includes flexible localized independence testing for non-stationary data, 
    allowing explicit regime shifts or uniform time-index chunking.
    '''
    def __init__(self,
                 data: np.ndarray,
                 groups: list[set[int]],
                 tau_max: int,
                 pc_alpha: float = 0.05,
                 max_conds_dim: int = 2,
                 u: Union[np.ndarray, str, None] = 'time_index',
                 conditional_independence_test: str = 'max_corr',
                 num_chunks_of_time_index: Union[int, None] = None,
                 pcmci_params: Union[dict[str, Any], None] = None,
                 non_stationarity_info: Optional[dict[str, Any]] = None,
                 verbose: int = 0,
                 **kwargs):
        
        super().__init__(data, groups, **kwargs)
        
        if conditional_independence_test not in conditional_independence_tests:
            raise ValueError(f"Unsupported independence test: {conditional_independence_test}")
        self.conditional_independence_test: ConditionalIndependence_base = conditional_independence_tests[conditional_independence_test]
        
        self._pcmci_params = pcmci_params if pcmci_params is not None else {}
        self.non_stationarity_info = non_stationarity_info or {}
        self._verbose = verbose
        
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.max_conds_dim = max_conds_dim
        
        self.device = self._get_device()
        self.u = None
        
        T_data = data.shape[0]

        # 1. Regime Construction / Time-Index Chunking
        if isinstance(u, str):
            if u == 'non_stationarity_shift':
                if self.non_stationarity_info.get('type') != 'regime_shifts':
                    raise ValueError("non_stationarity_info must have type 'regime_shifts' when u='non_stationarity_shift'")
                
                affected_vars = self.non_stationarity_info.get('affected_vars', [])
                if not affected_vars:
                    if self._verbose > 0:
                        logging.info("Notice: No variables affected by non-stationarity. Falling back to u='time_index'.")
                    u = 'time_index'
                else:
                    first_var = affected_vars[0]
                    shifts = self.non_stationarity_info['shift_details'][first_var]
                    
                    total_T = shifts[-1]['end']
                    u_full = np.zeros(total_T, dtype=int)
                    
                    for shift in shifts:
                        # Ensure 0-indexed mapping for regimes
                        regime_idx = shift['regime'] if shift['regime'] == 0 else shift['regime'] - 1
                        u_full[shift['start']:shift['end']] = regime_idx
                        
                    u_aligned = u_full[-T_data:]
                    num_regimes = self.non_stationarity_info.get('num_shifts', len(shifts)) + 1
                    
                    u_np = np.zeros((T_data, num_regimes))
                    u_np[np.arange(T_data), u_aligned] = 1
                    self.u = torch.tensor(u_np, dtype=torch.bool, device=self.device)

            if u == 'time_index':
                if num_chunks_of_time_index is None:
                    raise ValueError("num_chunks_of_time_index must be specified when u='time_index'")
                
                # Split uniformly across the time dimension
                chunk_indices = np.repeat(np.arange(num_chunks_of_time_index), 
                                          int(np.ceil(T_data / num_chunks_of_time_index)))[:T_data]
                u_np = np.zeros((T_data, num_chunks_of_time_index))
                u_np[np.arange(T_data), chunk_indices] = 1
                self.u = torch.tensor(u_np, dtype=torch.bool, device=self.device)
                
        elif isinstance(u, np.ndarray):
            self.u = torch.tensor(u, dtype=torch.bool, device=self.device)

        # 2. Pre-slice raw data into group tensors
        self._raw_group_data = [
            torch.tensor(self._data[:, list(group)], dtype=torch.float32, device=self.device) 
            for group in self._groups
        ]

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        # Not using mps due to potential non-implementation of certain operations in independence tests
        return torch.device('cpu')

    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        if self._verbose > 0:
            logging.info("Extracting parents: Group-PCMCI with localized tests.")
        final_parents, _ = self._run_group_pcmci()
        return final_parents

    def _test_ci(self, x_var: int, x_lag: int, y_var: int, y_lag: int, z_list: list[tuple[int, int]]) -> tuple[float, float]:
        T = self._raw_group_data[0].shape[0]
        start_t = 2 * self.tau_max 
        end_t = T
        
        if start_t >= end_t - 5:
            return 0.0, 1.0

        # Extract target groups
        X_data = self._raw_group_data[x_var][start_t - x_lag : end_t - x_lag].to(torch.float64)
        Y_data = self._raw_group_data[y_var][start_t - y_lag : end_t - y_lag].to(torch.float64)
        
        # Concatenate conditioning groups
        if z_list:
            Z_data_list = [self._raw_group_data[z_var][start_t - z_lag : end_t - z_lag].to(torch.float64) for z_var, z_lag in z_list]
            Z_data = torch.cat(Z_data_list, dim=1)
        else:
            Z_data = None

        # 3. Apply the Constructed Masks
        if self.u is not None:
            u_sliced = self.u[start_t : end_t]
            num_regimes = u_sliced.shape[1]
            regime_masks = [u_sliced[:, r] for r in range(num_regimes)]
        else:
            regime_masks = [torch.ones(end_t - start_t, dtype=torch.bool, device=self.device)]

        p_values = []
        stats = []

        # 4. Execute Independence Test per Regime/Chunk
        for mask in regime_masks:
            if mask.sum().item() < 20: 
                continue

            X_local = X_data[mask]
            Y_local = Y_data[mask]
            Z_local = Z_data[mask] if Z_data is not None else None

            if Z_local is not None:
                stat, pval = self.conditional_independence_test.conditional_test(X_local, Y_local, Z_local)
            else:
                stat, pval = self.conditional_independence_test.test(X_local, Y_local)
                
            pval = max(pval, 1e-15)
            p_values.append(pval)
            stats.append(stat)

        if not p_values:
            return 0.0, 1.0

        # 5. Fisher's Method to Combine the Localized Results
        chi2_stat = -2.0 * sum(math.log(p) for p in p_values)
        degrees_of_freedom = 2 * len(p_values)
        combined_p_value = float(chi2.sf(chi2_stat, degrees_of_freedom))
        mean_stat = sum(stats) / len(stats)

        return mean_stat, combined_p_value

    def _run_group_pcmci(self) -> tuple[dict[int, list[tuple[int, int]]], list]:
        N = len(self._groups)
        independencies_found = []
        parents = {j: [(i, tau) for i in range(N) for tau in range(1, self.tau_max + 1)] for j in range(N)}
        
        # Phase 1: PC1 Algorithm
        for j in range(N):
            p = 0
            while p <= self.max_conds_dim:
                candidate_parents = list(parents[j])
                to_remove = [] 
                for (i, tau) in candidate_parents:
                    available_conds = [c for c in parents[j] if c != (i, tau)]
                    if len(available_conds) < p:
                        continue
                    for Z in itertools.combinations(available_conds, p):
                        _, pval = self._test_ci(i, tau, j, 0, list(Z))
                        if pval > self.pc_alpha:
                            to_remove.append((i, tau))
                            independencies_found.append((i, tau, j, 0, list(Z)))
                            break
                for node in to_remove:
                    if node in parents[j]:
                        parents[j].remove(node)
                p += 1

        # Phase 2: MCI Algorithm
        final_parents = {j: [] for j in range(N)}
        for j in range(N):
            for (i, tau) in parents[j]:
                Z_j = [c for c in parents[j] if c != (i, tau)]
                Z_i = [(k, tau_k + tau) for (k, tau_k) in parents[i]]
                Z = list(set(Z_j + Z_i))
                
                _, pval = self._test_ci(i, tau, j, 0, Z)
                if pval > self.pc_alpha:
                    independencies_found.append((i, tau, j, 0, Z))
                else:
                    final_parents[j].append((i, -tau))
                    
        return final_parents, independencies_found