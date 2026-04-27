import numpy as np
import itertools
import math
from typing import Any, Optional, Union
from scipy.stats import chi2
import logging
import torch

from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.independence_tests import HSIC_Test
from group_causation.group_causal_discovery.iVAE.wrappers import IVAE_wrapper


class IVAE_GroupPCMCI_Proposal(GroupCausalDiscovery):
    '''
    Group causal discovery algorithm for non-stationary data.
    Uses iVAE for identifiable dimension reduction (using one-hot background u) on each group, 
    followed by a custom Group-PCMCI that uses HSIC to find causal links directly 
    between the group embeddings. Independence tests are localized per regime.
    '''
    def __init__(self, 
                 data: np.ndarray,
                 groups: list[set[int]],
                 u: Union[np.ndarray, str, None] = 'time_index',
                 num_chunks_of_time_index: Union[int, None] = None,
                 group_latent_dims_fraction: float = 0.33,
                 ivae_params: Union[dict[str, Any], None] = None,
                 pcmci_params: Union[dict[str, Any], None] = None,
                 non_stationarity_info: Optional[dict[str, Any]] = None,
                 verbose: int = 0,
                 **kwargs):
        
        super().__init__(data, groups, **kwargs)
        non_stationarity_info = non_stationarity_info if non_stationarity_info is not None else {}
        u = 'time_index' if u is None else u
        
        # ---------------------------------------------------------
        # 1. Background 'u' Construction (CPU - only runs once)
        # ---------------------------------------------------------
        if isinstance(u, str):
            if u == 'time_index':
                if num_chunks_of_time_index is None:
                    raise ValueError("num_chunks_of_time_index must be specified when u='time_index'")
                T_data = data.shape[0]
                chunk_indices = np.repeat(np.arange(num_chunks_of_time_index), int(np.ceil(T_data / num_chunks_of_time_index)))[:T_data]
                u_one_hot = np.zeros((T_data, num_chunks_of_time_index))
                u_one_hot[np.arange(T_data), chunk_indices] = 1
                self.u = u_one_hot
                
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
                    u_full[shift['start']:shift['end']] = shift['regime'] - 1
                
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
            
        raw_group_latent_dims = [group_latent_dims_fraction * len(group) for group in self._groups]
        if len(raw_group_latent_dims) != len(self._groups):
            raise ValueError("group_latent_dims must have one entry per group.")

        self._group_latent_dims = [
            max(1, min(int(np.ceil(dim)), len(group)))
            for dim, group in zip(raw_group_latent_dims, self._groups)
        ]
            
        self._ivae_params = ivae_params if ivae_params is not None else {}
        self._pcmci_params = pcmci_params if pcmci_params is not None else {}
        self._verbose = verbose
        
        self.tau_max = self._pcmci_params['tau_max']
        self.pc_alpha = self._pcmci_params['pc_alpha']
        self.max_conds_dim = self._pcmci_params['max_conds_dim']
        
        self.device = self._get_device()
        
        if isinstance(self.u, np.ndarray):
            self.u = torch.tensor(self.u, dtype=torch.float32, device=self.device)
        elif self.u is not None and not isinstance(self.u, torch.Tensor):
            self.u = torch.tensor(self.u, dtype=torch.float32, device=self.device)
        
        # ---------------------------------------------------------
        # 2. Extract Representations via iVAE
        # ---------------------------------------------------------
        self.group_embeddings = self._prepare_group_embeddings()

    def _get_device(self):
        if torch.cuda.is_available():
            logging.info("CUDA is available. Using GPU for computations.")
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            logging.info("MPS (Apple Silicon) is available. Using GPU for computations.")
            return torch.device('mps')
        logging.info("No GPU available. Using CPU for computations.")
        return torch.device('cpu')
    
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        if self._verbose > 0:
            logging.info("Extracting parents using custom Group-PCMCI with localized HSIC.")
        return self._run_group_pcmci()

    def _prepare_group_embeddings(self) -> list[torch.Tensor]:
        group_embeddings = []
        for idx, group in enumerate(self._groups):
            if self._verbose > 0:
                logging.info(f"Training iVAE for group {idx}")

            group_data = self._data[:, list(group)]
            
            group_params = dict(self._ivae_params)
            group_params.setdefault('inference_dim', self._group_latent_dims[idx])
            logging.info(f"iVAE input for group {idx}: {group_data.shape=}, {self.u.shape=}")
            group_latents, _, _, _ = IVAE_wrapper(group_data, self.u, device=self.device, **group_params)

            group_latents = group_latents.detach().clone().to(dtype=torch.float32, device=self.device)
            group_embeddings.append(group_latents)
            
        return group_embeddings

    def _test_ci(self, x_var: int, x_lag: int, y_var: int, y_lag: int, z_list: list[tuple[int, int]]) -> tuple[float, float]:
        T = self.group_embeddings[0].shape[0]
        start_t = 2 * self.tau_max 
        end_t = T
        
        if start_t >= end_t - 5:
            logging.warning("Not enough samples to test independence. Assuming independence.")
            return 0.0, 1.0

        X_data = self.group_embeddings[x_var][start_t - x_lag : end_t - x_lag].to(torch.float64)
        Y_data = self.group_embeddings[y_var][start_t - y_lag : end_t - y_lag].to(torch.float64)
        
        if z_list:
            Z_data_list = [self.group_embeddings[z_var][start_t - z_lag : end_t - z_lag].to(torch.float64) for z_var, z_lag in z_list]
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
                logging.warning(f"Regime with {num_samples} samples is too small. Skipping."
                                f"{u_sliced=}"
                                f"{regime_masks=}")
                continue

            X_local = X_data[mask]
            Y_local = Y_data[mask]

            if Z_data is not None:
                Z_local = Z_data[mask]
                
                stat, pval = HSIC_Test.conditional_test(
                    X_local, Y_local, Z_local, 
                    sequential_chunks=is_continuous,
                    max_samples=500 if is_continuous else X_local.shape[0] + 1
                )
            else:
                stat, pval = HSIC_Test.test(
                    X_local, Y_local, 
                    sequential_chunks=is_continuous,
                    max_samples=500 if is_continuous else X_local.shape[0] + 1
                )
            
            pval = max(pval, 1e-15)
            p_values.append(pval)
            stats.append(stat)

        if not p_values:
            return 0.0, 1.0

        # Aggregate using pure Python instead of NumPy to avoid transferring small lists
        chi2_stat = -2.0 * sum(math.log(p) for p in p_values)
        degrees_of_freedom = 2 * len(p_values)
        
        # Scipy's chi2 is CPU bound, but evaluating a single float is extremely fast and negligible
        combined_p_value = float(chi2.sf(chi2_stat, degrees_of_freedom))
        mean_stat = sum(stats) / len(stats)

        return mean_stat, combined_p_value

    def _run_group_pcmci(self) -> dict[int, list[tuple[int, int]]]:
        N = len(self._groups)
        
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
                        _, pval = self._test_ci(i, tau, j, 0, list(Z))
                        
                        if pval > self.pc_alpha:
                            to_remove.append((i, tau))
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
                
                stat, pval = self._test_ci(i, tau, j, 0, Z)
                
                if pval <= self.pc_alpha:
                    final_parents[j].append((i, -tau))
                    
        return final_parents