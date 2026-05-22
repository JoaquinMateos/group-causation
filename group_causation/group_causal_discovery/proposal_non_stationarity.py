import numpy as np
import itertools
import math
from typing import Any, Optional, Union
from scipy.stats import chi2
import logging
import torch

from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.dimensionality_reduction.iVAE.wrappers import IVAE_wrapper
from group_causation.independence_tests import conditional_independence_tests
from group_causation.independence_tests.conditional_independence_base import ConditionalIndependence_base

class IVAE_GroupPCMCI_Proposal(GroupCausalDiscovery):
    '''
    Group causal discovery algorithm for non-stationary data.
    Uses iVAE for identifiable dimension reduction (using one-hot background u) on each group, 
    followed by a custom Group-PCMCI that uses HSIC/MaxCorr to find causal links directly 
    between the group embeddings.
    '''
    def __init__(self,
                    data: np.ndarray,
                    groups: list[set[int]],
                    u: Union[np.ndarray, str, None] = 'time_index',
                    conditional_independence_test: str = 'max_corr',
                    num_chunks_of_time_index: Union[int, None] = None,
                    apply_adag_optimization: bool = False,
                    target_c_ind: float = 0.85,
                    fallback_latent_dims_fraction: float = 0.33,
                    ivae_params: Union[dict[str, Any], None] = None,
                    pcmci_params: Union[dict[str, Any], None] = None,
                    non_stationarity_info: Optional[dict[str, Any]] = None,
                    verbose: int = 0,
                    **kwargs):
            
            super().__init__(data, groups, **kwargs)
            non_stationarity_info = non_stationarity_info if non_stationarity_info is not None else {}
            u = 'time_index' if u is None else u
            
            if conditional_independence_test not in conditional_independence_tests:
                raise ValueError(f"Unsupported independence test: {conditional_independence_test}")
            self.conditional_independence_test: ConditionalIndependence_base = conditional_independence_tests[conditional_independence_test]
            
            # ---------------------------------------------------------
            # 1. Background 'u' Construction
            # ---------------------------------------------------------
            if isinstance(u, str):
                if u == 'non_stationarity_shift':
                    if non_stationarity_info.get('type') != 'regime_shifts':
                        raise ValueError("non_stationarity_info must have type 'regime_shifts' when u='non_stationarity_shift'")
                    
                    affected_vars = non_stationarity_info.get('affected_vars', [])
                    if not affected_vars:
                        # Fallback logic: If no variables are affected, shift u to 'time_index'
                        logging.info("Notice: No variables affected by non-stationarity. Falling back to u='time_index'.")
                        u = 'time_index'
                    else:
                        first_var = affected_vars[0]
                        shifts = non_stationarity_info['shift_details'][first_var]
                        
                        total_T = shifts[-1]['end']
                        u_full = np.zeros(total_T, dtype=int)
                        
                        for shift in shifts:
                            u_full[shift['start']:shift['end']] = shift['regime'] - 1
                        
                        T_data = data.shape[0]
                        u_aligned = u_full[-T_data:]
                        
                        num_regimes = non_stationarity_info.get('num_shifts', len(shifts)) + 1
                        u_one_hot = np.zeros((T_data, num_regimes))
                        u_one_hot[np.arange(T_data), u_aligned] = 1
                        
                        self.u = u_one_hot

                if u == 'time_index':
                    if num_chunks_of_time_index is None:
                        raise ValueError("num_chunks_of_time_index must be specified when u='time_index' (or when falling back to it)")
                    T_data = data.shape[0]
                    chunk_indices = np.repeat(np.arange(num_chunks_of_time_index), int(np.ceil(T_data / num_chunks_of_time_index)))[:T_data]
                    u_one_hot = np.zeros((T_data, num_chunks_of_time_index))
                    u_one_hot[np.arange(T_data), chunk_indices] = 1
                    self.u = u_one_hot
                    
                elif u != 'non_stationarity_shift':
                    raise ValueError(f"Unsupported value for u: {u}")
            else:
                self.u = u
                
            self._ivae_params = ivae_params if ivae_params is not None else {}
            self._pcmci_params = pcmci_params if pcmci_params is not None else {}
            self._verbose = verbose
            
            self.tau_max = self._pcmci_params.get('tau_max', 3)
            self.pc_alpha = self._pcmci_params.get('pc_alpha', 0.05)
            self.max_conds_dim = self._pcmci_params.get('max_conds_dim', 3)
            
            self.apply_adag_optimization = apply_adag_optimization
            self.target_c_ind = target_c_ind
            
            # Pre-slice raw data into groups for easy access during the c_ind checks
            self._raw_group_data = [
                torch.tensor(self._data[:, list(group)], dtype=torch.float32) 
                for group in self._groups
            ]
            
            # Calculate static fallback dimensions in case Adag is disabled
            self._fallback_dims = [
                max(1, min(int(np.ceil(fallback_latent_dims_fraction * len(group))), len(group)))
                for group in self._groups
            ]
                
            self.device = self._get_device()
            
            if isinstance(self.u, np.ndarray):
                self.u = torch.tensor(self.u, dtype=torch.float32, device=self.device)
            elif self.u is not None and not isinstance(self.u, torch.Tensor):
                self.u = torch.tensor(self.u, dtype=torch.float32, device=self.device)
            
            for i in range(len(self._raw_group_data)):
                self._raw_group_data[i] = self._raw_group_data[i].to(self.device)

    def _get_device(self):
        if torch.cuda.is_available():
            logging.debug("CUDA is available. Using GPU for computations.")
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            logging.debug("MPS (Apple Silicon) is available. Using GPU for computations.")
            return torch.device('mps')
        logging.debug("No GPU available. Using CPU for computations.")
        return torch.device('cpu')
    
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        """Main entry point. Branches depending on the apply_adag_optimization hyperparameter."""
        if self.apply_adag_optimization:
            if self._verbose > 0:
                logging.info(f"Extracting parents: Adag optimization ENABLED (target c_ind={self.target_c_ind}).")
            return self._run_adag()
        else:
            if self._verbose > 0:
                logging.info(f"Extracting parents: Adag optimization DISABLED. Using static dimensions {self._fallback_dims}.")
            group_embeddings = self._prepare_group_embeddings(self._fallback_dims)
            final_parents, _ = self._run_group_pcmci(group_embeddings)
            return final_parents

    def _run_adag(self) -> dict[int, list[tuple[int, int]]]:
        """Iterative aggregation loop to find the optimal latent dimensionality."""
        N = len(self._groups)
        m = [1] * N
        max_m = [len(group) for group in self._groups]
        
        c_ind_score = 0.0
        final_parents = {}
        
        while c_ind_score < self.target_c_ind:
            logging.info(f"Adag Iteration | Current latent dimensions m: {m}")
                
            group_embeddings = self._prepare_group_embeddings(m)
            parents, independencies = self._run_group_pcmci(group_embeddings)
            
            c_ind_score = self._compute_c_ind(independencies)
            
            logging.info(f"Target c_ind: {self.target_c_ind} | Achieved c_ind: {c_ind_score:.3f}")
            
            final_parents = parents
            
            if c_ind_score >= self.target_c_ind or m == max_m:
                logging.info("Adag termination condition met.")
                break
                
            for i in range(N):
                if m[i] < max_m[i]:
                    m[i] += 1
                    
        return final_parents

    def _compute_c_ind(self, independencies: list[tuple[int, int, int, int, list[tuple[int, int]]]]) -> float:
        """Evaluates independence consistency against the raw un-aggregated data."""
        if not independencies:
            return 1.0 
            
        C_ind_count = 0
        I_ind_count = 0
        
        for (x_var, x_lag, y_var, y_lag, z_list) in independencies:
            _, pval = self._test_ci(self._raw_group_data, x_var, x_lag, y_var, y_lag, z_list)
            
            if pval > self.pc_alpha:
                C_ind_count += 1
            else:
                I_ind_count += 1
                
        total = C_ind_count + I_ind_count
        return C_ind_count / total if total > 0 else 1.0

    def _prepare_group_embeddings(self, latent_dims: list[int]) -> list[torch.Tensor]:
        group_embeddings = []
        for idx, group in enumerate(self._groups):
            group_data = self._data[:, list(group)]
            
            group_params = dict(self._ivae_params)
            group_params['inference_dim'] = latent_dims[idx]
            
            group_latents, _, _, _ = IVAE_wrapper(group_data, self.u, device=self.device, **group_params)
            group_latents = group_latents.detach().clone().to(dtype=torch.float32, device=self.device)
            group_embeddings.append(group_latents)
            
        return group_embeddings

    def _test_ci(self, data_source: list[torch.Tensor], x_var: int, x_lag: int, y_var: int, y_lag: int, z_list: list[tuple[int, int]]) -> tuple[float, float]:
        T = data_source[0].shape[0]
        start_t = 2 * self.tau_max 
        end_t = T
        
        if start_t >= end_t - 5:
            logging.warning("Not enough samples to test independence. Assuming independence.")
            return 0.0, 1.0

        X_data = data_source[x_var][start_t - x_lag : end_t - x_lag].to(torch.float64)
        Y_data = data_source[y_var][start_t - y_lag : end_t - y_lag].to(torch.float64)
        
        if z_list:
            Z_data_list = [data_source[z_var][start_t - z_lag : end_t - z_lag].to(torch.float64) for z_var, z_lag in z_list]
            Z_data = torch.cat(Z_data_list, dim=1)
        else:
            Z_data = None

        u_sliced = self.u[start_t : end_t]
        
        if u_sliced.ndim == 2:
            unique_vals = torch.unique(u_sliced, dim=0)
        else:
            unique_vals = torch.unique(u_sliced)
            
        is_continuous = len(unique_vals) > (len(u_sliced) // 2)

        if is_continuous:
            regime_masks = [torch.ones(len(u_sliced), dtype=torch.bool, device=self.device)]
        elif u_sliced.ndim == 2 and torch.all((u_sliced == 0) | (u_sliced == 1)): 
            num_regimes = u_sliced.shape[1]
            regime_masks = [u_sliced[:, r] == 1 for r in range(num_regimes)]
        else: 
            if u_sliced.ndim == 2:
                unique_rows = torch.unique(u_sliced, dim=0)
                regime_masks = [torch.all(u_sliced == row, dim=1) for row in unique_rows]
            else:
                regime_masks = [u_sliced == val for val in torch.unique(u_sliced)]

        p_values = []
        stats = []
        
        for mask in regime_masks:
            num_samples = mask.sum().item()
            if num_samples < 20:
                continue

            X_local = X_data[mask]
            Y_local = Y_data[mask]

            if Z_data is not None:
                Z_local = Z_data[mask]
                stat, pval = self.conditional_independence_test.conditional_test(
                    X_local, Y_local, Z_local, 
                    sequential_chunks=is_continuous,
                    max_samples=500 if is_continuous else X_local.shape[0] + 1
                )
            else:
                stat, pval = self.conditional_independence_test.test(
                    X_local, Y_local, 
                    sequential_chunks=is_continuous,
                    max_samples=500 if is_continuous else X_local.shape[0] + 1
                )
            
            pval = max(pval, 1e-15)
            p_values.append(pval)
            stats.append(stat)

        if not p_values:
            return 0.0, 1.0

        chi2_stat = -2.0 * sum(math.log(p) for p in p_values)
        degrees_of_freedom = 2 * len(p_values)
        combined_p_value = float(chi2.sf(chi2_stat, degrees_of_freedom))
        mean_stat = sum(stats) / len(stats)

        return mean_stat, combined_p_value

    def _run_group_pcmci(self, group_embeddings: list[torch.Tensor]) -> tuple[dict[int, list[tuple[int, int]]], list]:
        N = len(self._groups)
        independencies_found = []
        
        # Phase 1: PC1 Algorithm
        parents = {j: [(i, tau) for i in range(N) for tau in range(1, self.tau_max + 1)] for j in range(N)}
        
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
                        _, pval = self._test_ci(group_embeddings, i, tau, j, 0, list(Z))
                        
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
                
                _, pval = self._test_ci(group_embeddings, i, tau, j, 0, Z)
                
                if pval > self.pc_alpha:
                    independencies_found.append((i, tau, j, 0, Z))
                else:
                    final_parents[j].append((i, -tau))
                    
        return final_parents, independencies_found