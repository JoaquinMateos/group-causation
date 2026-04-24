import numpy as np
import itertools
from typing import Any, Optional, Union
from sklearn.linear_model import LinearRegression
import logging

from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.group_causal_discovery.group_resit import HSIC_Test
from group_causation.group_causal_discovery.iVAE.wrappers import IVAE_wrapper

# Assuming HSIC_Test is imported from your library
# from your_library import HSIC_Test 

class IVAE_GroupPCMCI_Proposal(GroupCausalDiscovery):
    '''
    Group causal discovery algorithm for non-stationary data.
    Uses iVAE for identifiable dimension reduction (using one-hot background u) on each group, 
    followed by a custom Group-PCMCI that uses HSIC to find causal links directly 
    between the group embeddings.
    '''
    def __init__(self, 
                 data: np.ndarray,
                 groups: list[set[int]],
                 u: Union[np.ndarray, str, None] = 'time_index',
                 group_latent_dims: Union[int, list[int]] = 2,
                 ivae_params: Union[dict[str, Any], None] = None,
                 pcmci_params: Union[dict[str, Any], None] = None,
                 non_stationarity_info: Optional[dict[str, Any]] = None,
                 verbose: int = 0,
                 **kwargs):
        
        super().__init__(data, groups, **kwargs)
        non_stationarity_info = non_stationarity_info if non_stationarity_info is not None else {}
        u = 'time_index' if u is None else u
        
        # ---------------------------------------------------------
        # 1. Background 'u' Construction
        # ---------------------------------------------------------
        if isinstance(u, str):
            if u == 'time_index':
                self.u = np.arange(data.shape[0]).reshape(-1, 1)
            elif u == 'non_stationarity_shift':
                if non_stationarity_info.get('type') != 'regime_shifts':
                    raise ValueError("non_stationarity_info must have type 'regime_shifts' when u='non_stationarity_shift'")
                
                affected_vars = non_stationarity_info.get('affected_vars', [])
                if not affected_vars:
                    raise ValueError("No variables were affected by non-stationarity, cannot build 'u'.")
                
                first_var = affected_vars[0]
                shifts = non_stationarity_info['shift_details'][first_var]
                
                total_T = shifts[-1]['end']
                u_full = np.zeros(total_T, dtype=int)
                
                for shift in shifts:
                    u_full[shift['start']:shift['end']] = shift['regime']
                
                T_data = data.shape[0]
                u_aligned = u_full[-T_data:]
                
                num_regimes = non_stationarity_info.get('num_shifts', len(shifts)) + 1
                u_one_hot = np.zeros((T_data, num_regimes))
                u_one_hot[np.arange(T_data), u_aligned] = 1
                
                self.u = u_one_hot
            else:
                raise ValueError(f"Unsupported value for u: {u}")
        else:
            self.u = u
            
        self._group_latent_dims = [group_latent_dims] * len(self._groups) if isinstance(group_latent_dims, int) else group_latent_dims
        if len(self._group_latent_dims) != len(self._groups):
            raise ValueError("group_latent_dims must have one entry per group.")
            
        self._ivae_params = ivae_params if ivae_params is not None else {}
        self._pcmci_params = pcmci_params if pcmci_params is not None else {}
        self._verbose = verbose
        
        # PCMCI Default settings
        self.tau_max = self._pcmci_params.get('tau_max', 1)
        self.pc_alpha = self._pcmci_params.get('pc_alpha', 0.05)
        self.max_conds_dim = self._pcmci_params.get('max_conds_dim', 3)
        
        # ---------------------------------------------------------
        # 2. Extract Representations via iVAE
        # ---------------------------------------------------------
        self.group_embeddings = self._prepare_group_embeddings()

    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        '''
        Run the custom Group-PCMCI algorithm on the iVAE embeddings.
        '''
        if self._verbose > 0:
            logging.info("Extracting parents using custom Group-PCMCI with Residual-HSIC.")
        return self._run_group_pcmci()

    def _prepare_group_embeddings(self) -> list[np.ndarray]:
        '''
        Reduces dimensionality of each group. Returns a list of embeddings.
        '''
        group_embeddings = []
        for idx, group in enumerate(self._groups):
            if self._verbose > 0:
                logging.info(f"Training iVAE for group {idx}")

            group_data = self._data[:, list(group)]
            
            # Reduce dimensionality
            group_params = dict(self._ivae_params)
            group_params.setdefault('inference_dim', self._group_latent_dims[idx])
            group_latents, _, _, _ = IVAE_wrapper(group_data, self.u, **group_params)

            # Append the group representations directly
            group_embeddings.append(group_latents)
            
        return group_embeddings

    def _test_ci(self, x_var: int, x_lag: int, y_var: int, y_lag: int, z_list: list[tuple[int, int]]) -> tuple[float, float]:
        '''
        Evaluates Momentary Conditional Independence using Residual HSIC.
        $X_{t-tau} \perp\!\!\!\perp Y_t \mid Z$
        '''
        all_lags = [x_lag, y_lag] + [z_lag for _, z_lag in z_list]
        max_l = max(all_lags) if all_lags else 0
        
        T = self.group_embeddings[0].shape[0]
        start_t = max_l
        end_t = T
        
        # If the required lag exceeds our time series length, safely assume independent
        if start_t >= end_t - 5: 
            return 0.0, 1.0

        # Extract aligned time series segments
        X_data = self.group_embeddings[x_var][start_t - x_lag : end_t - x_lag]
        Y_data = self.group_embeddings[y_var][start_t - y_lag : end_t - y_lag]
        
        if z_list:
            Z_data_list = []
            for z_var, z_lag in z_list:
                Z_data_list.append(self.group_embeddings[z_var][start_t - z_lag : end_t - z_lag])
            
            Z_data = np.concatenate(Z_data_list, axis=1)
            
            # Partial out Z
            reg_x = LinearRegression().fit(Z_data, X_data)
            res_x = X_data - reg_x.predict(Z_data)
            
            reg_y = LinearRegression().fit(Z_data, Y_data)
            res_y = Y_data - reg_y.predict(Z_data)
        else:
            res_x = X_data
            res_y = Y_data
            
        return HSIC_Test.test(res_x, res_y)

    def _run_group_pcmci(self) -> dict[int, list[tuple[int, int]]]:
        '''
        Implements a streamlined PCMCI logic natively.
        Phase 1: PC1 Condition Selection
        Phase 2: MCI Evaluation
        '''
        N = len(self._groups)
        
        # ---------------------------------------------------------
        # Phase 1: PC1 Algorithm (Identify candidate superset of parents)
        # ---------------------------------------------------------
        # Initialize fully connected pasts
        parents = {j: [(i, tau) for i in range(N) for tau in range(1, self.tau_max + 1)] for j in range(N)}
        
        for j in range(N):
            p = 0
            while p <= self.max_conds_dim:
                candidate_parents = list(parents[j])
                
                for (i, tau) in candidate_parents:
                    available_conds = [c for c in parents[j] if c != (i, tau)]
                    
                    if len(available_conds) < p:
                        continue
                        
                    # Test combinations of condition size `p`
                    for Z in itertools.combinations(available_conds, p):
                        _, pval = self._test_ci(i, tau, j, 0, list(Z))
                        
                        if pval > self.pc_alpha:
                            parents[j].remove((i, tau))
                            break # Link removed, move to next candidate parent
                p += 1

        # ---------------------------------------------------------
        # Phase 2: MCI Algorithm (Momentary Conditional Independence)
        # ---------------------------------------------------------
        final_parents = {j: [] for j in range(N)}
        
        for j in range(N):
            for (i, tau) in parents[j]:
                # Conditioning set: P(j) \ {(i, tau)} U P(i) shifted by tau
                Z_j = [c for c in parents[j] if c != (i, tau)]
                
                # Shift parents of 'i' by tau. Only keep if shift is within bounds
                Z_i = [(k, tau_k + tau) for (k, tau_k) in parents[i]]
                
                # Deduplicate conditions
                Z = list(set(Z_j + Z_i))
                
                stat, pval = self._test_ci(i, tau, j, 0, Z)
                
                # If dependent under maximum conditioning, accept the causal link
                if pval <= self.pc_alpha:
                    final_parents[j].append((i, -tau))
                    
        return final_parents