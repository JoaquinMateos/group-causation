from types import MappingProxyType

import numpy as np
import itertools
import math
from typing import Any, Mapping, Optional, Union
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
            
            self.tau_min = self._pcmci_params.get('tau_min', 1)
            self.tau_max = self._pcmci_params['tau_max']
            self.pc_alpha = self._pcmci_params['pc_alpha']
            self.max_conds_dim = self._pcmci_params['max_conds_dim']
            
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
    
    def extract_parents(self, forbidden_parents: Mapping[int, list[tuple[int, int]]] = MappingProxyType({})) -> dict[int, list[tuple[int, int]]]:
        """
        Main entry point. Branches depending on the apply_adag_optimization hyperparameter.
        
        Args:
            forbidden_parents: Dictionary mapping a target group index to a list of forbidden (parent_index, lag) tuples.
                               Example: {4: [(5, 0)]} means "Group 5 at lag 0 cannot cause target Group 4".
        """
        if self.apply_adag_optimization:
            if self._verbose > 0:
                logging.info(f"Extracting parents: Adag optimization ENABLED (target c_ind={self.target_c_ind}).")
            return self._run_adag(forbidden_parents=forbidden_parents)
        else:
            if self._verbose > 0:
                logging.info(f"Extracting parents: Adag optimization DISABLED. Using static dimensions {self._fallback_dims}.")
            group_embeddings = self._prepare_group_embeddings(self._fallback_dims)
            final_parents, _ = self._run_group_pcmci(group_embeddings, forbidden_parents=forbidden_parents)
            return final_parents

    def _run_adag(self, forbidden_parents: Mapping[int, list[tuple[int, int]]] = MappingProxyType({})) -> dict[int, list[tuple[int, int]]]:
        """Iterative aggregation loop to find the optimal latent dimensionality."""
        N = len(self._groups)
        m = [min(1, len(group)) for group in self._groups]
        max_m = [len(group) for group in self._groups]
        
        c_ind_score = 0.0
        final_parents = {}
        
        while c_ind_score < self.target_c_ind:
            logging.info(f"Adag Iteration | Current latent dimensions m: {m}")
                
            group_embeddings = self._prepare_group_embeddings(m)
            parents, independencies = self._run_group_pcmci(group_embeddings, forbidden_parents=forbidden_parents)
            
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

        # 1. Data and conditioning variable extraction
        X_data = data_source[x_var][start_t - x_lag : end_t - x_lag].to(torch.float64)
        Y_data = data_source[y_var][start_t - y_lag : end_t - y_lag].to(torch.float64)
        
        if z_list:
            Z_data_list = [data_source[z_var][start_t - z_lag : end_t - z_lag].to(torch.float64) for z_var, z_lag in z_list]
            Z_data = torch.cat(Z_data_list, dim=1)
        else:
            Z_data = None

        # 2. Analysis of background variable 'u' for regime detection or continuous handling
        u_sliced = self.u[start_t : end_t]
        
        if u_sliced.ndim == 2:
            unique_vals = torch.unique(u_sliced, dim=0)
        else:
            unique_vals = torch.unique(u_sliced)
            
        is_continuous = len(unique_vals) > (len(u_sliced) // 2)

        # ---------------------------------------------------------------------
        # 3. Delegation to Test (Continuous 'u' Handling)
        # ---------------------------------------------------------------------
        if is_continuous:
            # Delegate the partitioning and testing to the independence test class,
            # which should handle continuous 'u' appropriately (e.g., by using it
            # as a covariate or through other means depending on the test's capabilities).
            if Z_data is not None:
                return self.conditional_independence_test.conditional_test(
                    X_data, Y_data, Z_data, max_samples=500, sequential_chunks=True
                )
            else:
                return self.conditional_independence_test.test(
                    X_data, Y_data, max_samples=500, sequential_chunks=True
                )

        # ---------------------------------------------------------------------
        # 4. Delegation to Test (Regime-wise Testing)
        # ---------------------------------------------------------------------
        if u_sliced.ndim == 2 and torch.all((u_sliced == 0) | (u_sliced == 1)): 
            num_regimes = u_sliced.shape[1]
            regime_masks = [u_sliced[:, r] == 1 for r in range(num_regimes)]
        else: 
            if u_sliced.ndim == 2:
                unique_rows = torch.unique(u_sliced, dim=0)
                regime_masks = [torch.all(u_sliced == row, dim=1) for row in unique_rows]
            else:
                regime_masks = [u_sliced == val for val in torch.unique(u_sliced)]

        X_regimes, Y_regimes, Z_regimes = [], [], []
        
        for mask in regime_masks:
            # Minimum samples check for reliable testing (especially for OLS-based tests that require variance estimation)
            if mask.sum().item() < 6:
                logging.warning(f"Regime with {mask.sum().item()} samples is too small for reliable testing. Skipping this regime.")
                continue
            X_regimes.append(X_data[mask])
            Y_regimes.append(Y_data[mask])
            if Z_data is not None:
                Z_regimes.append(Z_data[mask])

        if not X_regimes:
            return 0.0, 1.0

        # Optimal route: Use native regime-wise CCT methods if supported by the test (like MaxCorr_Test)
        if hasattr(self.conditional_independence_test, 'conditional_test_regimes'):
            if Z_data is not None:
                return self.conditional_independence_test.conditional_test_regimes(X_regimes, Y_regimes, Z_regimes)
            else:
                return self.conditional_independence_test.test_regimes(X_regimes, Y_regimes)
        
        # General fallback if the test does not support regime-wise testing:
        # perform individual tests and aggregate with CCT
        else:
            logging.warning("The specified independence test does not support regime-wise testing. Falling back to individual tests with CCT aggregation.")
            stats, p_vals, weights = [], [], []
            valid_samples = sum(x.shape[0] for x in X_regimes)
            
            for i in range(len(X_regimes)):
                if Z_data is not None:
                    s, p = self.conditional_independence_test.conditional_test(X_regimes[i], Y_regimes[i], Z_regimes[i])
                else:
                    s, p = self.conditional_independence_test.test(X_regimes[i], Y_regimes[i])
                
                p = max(1e-15, min(1.0 - 1e-15, p))
                stats.append(s)
                p_vals.append(p)
                weights.append(X_regimes[i].shape[0] / valid_samples)
                
            if not p_vals:
                return 0.0, 1.0
                
            t_stat = sum(w * math.tan(math.pi * (0.5 - p)) for w, p in zip(weights, p_vals))
            global_p_val = 0.5 - (math.atan(t_stat) / math.pi)
            avg_stat = sum(w * s for w, s in zip(weights, stats))
            return avg_stat, global_p_val

    def _run_group_pcmci(self, group_embeddings: list[torch.Tensor],
                         forbidden_parents: Mapping[int, list[tuple[int, int]]] = MappingProxyType({})) -> tuple[dict[int, list[tuple[int, int]]], list]:
        N = len(self._groups)
        independencies_found = []
        
        # Inicializamos la matriz de efectos
        self.effect_val_matrix = np.zeros((N, N, self.tau_max + 1))
        
        # Phase 1: PC1 Algorithm
        parents = {}
        for j in range(N):
            parents[j] = [
                (i, tau) 
                for i in range(N) 
                for tau in range(self.tau_min, self.tau_max + 1)
                if not (tau == 0 and i == j) # Skip lag-0 self-loops
                and (i, tau) not in forbidden_parents.get(j, []) # Skip forbidden domain-knowledge edges
            ]
        
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
                        # En la Fase 1 solo nos importa el p-valor para podar
                        _, pval = self._test_ci(group_embeddings, i, tau, j, 0, list(Z))
                        
                        if pval > self.pc_alpha:
                            independencies_found.append((i, tau, j, 0, Z))
                            to_remove.append((i, tau))
                            break # Found independence, no need to test more conditioning sets for this candidate parent
                            
                # Remove all parents that were found to be independent in this iteration
                parents[j] = [c for c in parents[j] if c not in to_remove]
                
                # Optimization: if no parents left or we have already exceeded the max conditioning dimension, we can stop early for this target
                if not parents[j] or p >= len(parents[j]):
                    break
                    
                p += 1

        # Phase 2: MCI Algorithm
        final_parents = {j: [] for j in range(N)}
        
        for j in range(N):
            for (i, tau) in parents[j]:
                Z_j = [c for c in parents[j] if c != (i, tau)]
                Z_i = [(k, tau_k + tau) for (k, tau_k) in parents[i]]
                
                Z = list(set(Z_j + Z_i))
                
                # Recuperamos el estadístico 'stat' en la Fase 2
                stat, pval = self._test_ci(group_embeddings, i, tau, j, 0, Z)
                
                if pval > self.pc_alpha:
                    independencies_found.append((i, tau, j, 0, Z))
                else:
                    final_parents[j].append((i, -tau))
                    
                    # Llenamos la matriz con los valores finales del estadístico
                    self.effect_val_matrix[i, j, tau] = abs(stat)
                    
                    # Relaciones instantáneas deben ser simétricas en la matriz
                    if tau == 0:
                        self.effect_val_matrix[j, i, 0] = abs(stat)
                    
        return final_parents, independencies_found
    
    def get_effect_val_matrix(self) -> np.ndarray:
        """Returns the matrix of effect values computed during the Group-PCMCI phase."""
        if not hasattr(self, 'effect_val_matrix'):
            raise ValueError("Effect value matrix is not available. Please run extract_parents() first to compute it.")
        return self.effect_val_matrix