import logging
import math
from sklearn.decomposition import PCA
import torch
import numpy as np
from typing import Any, Callable, List, Tuple, Optional, Dict
from scipy.stats import chi2

from group_causation.dimensionality_reduction.iVAE.wrappers import IVAEDimensionalityReduction

class TunableDeepLatent:
    """Tunable aggregation map using your provided VAE or iVAE wrappers."""
    
    def __init__(self, **model_kwargs):
        """
        Args:
            **model_kwargs: Arguments to pass to the reducer (e.g., max_iter, lr, device, batch_size).
        """
        self.model_kwargs = model_kwargs

    def aggregate(self, X: torch.Tensor, m: int, U: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reduces vector variable X to a latent representation of dimension m.
        """
        dim = X.shape[1] if X.ndim > 1 else 1
        
        # Cap the latent dimension at the original data dimension to prevent unwanted expansion
        m = min(m, dim)
        
        if U is None:
            raise ValueError("Auxiliary tensor 'U' must be provided when use_auxiliary=True (iVAE).")
        # Force both latent_dim and inference_dim to m
        reducer = IVAEDimensionalityReduction(latent_dim=m, **self.model_kwargs)
        return reducer.fit_transform(X, U)

class TunablePCA:
    """Tunable aggregation map using PCA to interface with AdagWrapper."""
    def aggregate(self, X: torch.Tensor, m: int, U: Optional[torch.Tensor] = None) -> torch.Tensor:
        dim = X.shape[1] if X.ndim > 1 else 1
        m = min(m, dim)
        
        pca = PCA(n_components=m)
        X_np = X.cpu().numpy()
        X_pca = pca.fit_transform(X_np)
        
        return torch.tensor(X_pca, dtype=torch.float32, device=X.device)

class AdagWrapper:
    """
    Adaptive Aggregation (Adag) wrapper for Causal Discovery over vector-valued variables.
    Includes localized conditional independence testing for non-stationary time series.
    """
    def __init__(self, ci_test_class, groups: List[List[int]], max_lag: int,
                 p_val_threshold: float = 0.05, num_regimes: int = 1):
        self.ci_test = ci_test_class
        self._groups = groups
        self.max_lag = max_lag
        self.alpha = p_val_threshold
        self.num_regimes = num_regimes
        self._raw_group_data = None # Will be populated during run()

    def is_independent(self, p_val: float) -> bool:
        return p_val > self.alpha

    def _compute_c_ind(self, group_parents: Dict[int, List[Tuple[int, int]]]) -> float:
        """
        Evaluates independence consistency on the raw un-aggregated data.
        """
        C_ind_count = 0
        I_ind_count = 0
        N = len(self._groups)
        
        for j in range(N):
            parents_of_j = group_parents.get(j, [])
            
            for i in range(N):
                for tau in range(1, self.max_lag + 1):
                    if (i, -tau) not in parents_of_j:
                        
                        pval = self._test_ci_raw(
                            x_var=i, x_lag=tau, 
                            y_var=j, y_lag=0, 
                            cond_groups=parents_of_j
                        )
                        
                        if self.is_independent(pval):
                            C_ind_count += 1
                        else:
                            I_ind_count += 1
                        
        total = C_ind_count + I_ind_count
        return C_ind_count / total if total > 0 else 1.0

    def _test_ci_raw(self, x_var: int, x_lag: int, y_var: int, y_lag: int, cond_groups: List[Tuple[int, int]]) -> float:
        """
        Tests conditional independence between raw high-dimensional groups.
        Splits the aligned time-series into `num_regimes` chunks to handle non-stationarity.
        """
        if self._raw_group_data is None:
            raise RuntimeError("Raw group data is missing. AdagWrapper.run() must be called first.")
            
        T = self._raw_group_data[0].shape[0]
        
        max_z_lag = max([abs(lag) for _, lag in cond_groups]) if cond_groups else 0
        safe_max_lag = max(x_lag, y_lag, max_z_lag)
        
        start_t = safe_max_lag
        end_t = T
        
        if start_t >= end_t - 5:
            return 1.0 
            
        # 1. Shift target variables
        X_full = self._raw_group_data[x_var][start_t - x_lag : end_t - x_lag]
        Y_full = self._raw_group_data[y_var][start_t - y_lag : end_t - y_lag]
        
        Z_full = None
        if cond_groups:
            Z_list = [
                self._raw_group_data[z_var][start_t - abs(z_lag) : end_t - abs(z_lag)] 
                for z_var, z_lag in cond_groups
            ]
            Z_full = torch.cat(Z_list, dim=1)
            
        # 2. Localized Testing (Chunking)
        chunk_size = int(np.ceil(X_full.shape[0] / self.num_regimes))
        p_values = []
        
        for r in range(self.num_regimes):
            idx_start = r * chunk_size
            idx_end = min((r + 1) * chunk_size, X_full.shape[0])
            
            # Skip chunks that are too small for a valid statistical test
            if idx_end - idx_start < 20:
                continue
                
            X_chunk = X_full[idx_start:idx_end]
            Y_chunk = Y_full[idx_start:idx_end]
            
            if Z_full is None:
                _, pval = self.ci_test.test(X_chunk, Y_chunk)
            else:
                Z_chunk = Z_full[idx_start:idx_end]
                if hasattr(self.ci_test, 'conditional_test'):
                    _, pval = self.ci_test.conditional_test(X_chunk, Y_chunk, Z_chunk)
                else:
                    _, pval = self.ci_test.test(X_chunk, Y_chunk, Z_chunk)
                    
            p_values.append(max(pval, 1e-15))
            
        # 3. Combine P-values
        if not p_values:
            return 1.0
            
        if self.num_regimes == 1:
            return p_values[0]
            
        # Fisher's Method for combining independent p-values
        chi2_stat = -2.0 * sum(math.log(p) for p in p_values)
        degrees_of_freedom = 2 * len(p_values)
        combined_p_value = float(chi2.sf(chi2_stat, degrees_of_freedom))
        
        return combined_p_value

    def run(self, 
            X_data: List[torch.Tensor], 
            discovery_func: Callable, 
            aggregator: Any,
            U_data: Optional[List[torch.Tensor]] = None,
            target_alpha_q: float = 0.8,
            **kwargs) -> Tuple[List[torch.Tensor], float, List[int]]:
        
        self._raw_group_data = X_data
        
        N = len(X_data)
        m = [1] * N
        max_m = [x.shape[1] if x.ndim > 1 else 1 for x in X_data]
        
        c_ind_score = 0.0
        Z_m = []

        while c_ind_score < target_alpha_q:
            logging.info(f"--- Adag Iteration | Current dimensions m: {m} ---")
            
            Z_m = []
            for i in range(N):
                U_i = U_data[i] if U_data is not None else None
                Z_m.append(aggregator.aggregate(X_data[i], m[i], U=U_i))
            
            group_parents = discovery_func(Z_m)
            logging.info(f"Discovered independencies at m={m}: {group_parents}")
            
            c_ind_score = self._compute_c_ind(group_parents)
            logging.info(f"Target c_ind: {target_alpha_q} | Achieved c_ind: {c_ind_score:.3f}")
            
            if c_ind_score >= target_alpha_q or m == max_m:
                break
                
            for i in range(N):
                if m[i] < max_m[i]:
                    m[i] += 1
                    
        return Z_m, c_ind_score, m