from abc import ABC, abstractmethod
import logging
import torch
import numpy as np
from sklearn.decomposition import PCA
from typing import Any, Callable, Type

from group_causation.dimensionality_reduction.dimensionality_reduction_base import DimensionalityReduction
from group_causation.dimensionality_reduction.iVAE.wrappers import IVAEWrapper
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.independence_tests.conditional_independence_base import ConditionalIndependence_base
from group_causation.causal_discovery_base import CausalDiscovery # Asegúrate de importar GroupCausalDiscovery donde corresponda


class AggregationMap(ABC):
    """Abstract base class for aggregation maps."""

    @abstractmethod
    def aggregate(self, X: torch.Tensor, m: int, U: torch.Tensor | None = None) -> torch.Tensor:
        """
        Reduces vector variable X to a latent representation of dimension m.
        
        Args:
            X (torch.Tensor): Input tensor of shape (T, d) where T is the number of samples and d is the original dimension.
            m (int): Target latent dimension for the aggregated representation.
            U (torch.Tensor | None): Optional auxiliary tensor for iVAE-based aggregation.
        
        Returns:
            torch.Tensor: Aggregated representation of shape (T, m).
        """
        raise NotImplementedError("Subclasses must implement the aggregate method.")

class TunableDeepLatent(AggregationMap):
    """Tunable aggregation map using your provided VAE or iVAE wrappers."""
    
    def __init__(self, **model_kwargs):
        """
        Args:
            **model_kwargs: Arguments to pass to the reducer (e.g., max_epoch, lr, device, batch_size).
        """
        self.model_kwargs = model_kwargs

    def aggregate(self, X: torch.Tensor, m: int, U: torch.Tensor | None = None) -> torch.Tensor:
        """
        Reduces vector variable X to a latent representation of dimension m.
        """
        dim = X.shape[1] if X.ndim > 1 else 1
        
        # Cap the latent dimension at the original data dimension to prevent unwanted expansion
        m = min(m, dim)
        
        if U is None:
            raise ValueError("Auxiliary tensor 'U' must be provided when use_auxiliary=True (iVAE).")
        
        # Instantiate and fit the new class-based IVAEWrapper using the unified interface
        reducer = IVAEWrapper(latent_dim=m, **self.model_kwargs)
        return reducer.fit_transform(X, U)


class TunablePCA(AggregationMap):
    """Tunable aggregation map using PCA to interface with AdagWrapper."""
    
    def aggregate(self, X: torch.Tensor, m: int, U: torch.Tensor | None = None) -> torch.Tensor:
        dim = X.shape[1] if X.ndim > 1 else 1
        m = min(m, dim)
        
        pca = PCA(n_components=m)
        X_np = X.cpu().numpy()
        X_pca = pca.fit_transform(X_np)
        
        return torch.tensor(X_pca, dtype=torch.float32, device=X.device)


class AdagWrapper(DimensionalityReduction):
    """
    Adaptive Aggregation (Adag) wrapper for Causal Discovery over vector-valued variables.
    """
    def __init__(self, 
                 ci_test_class: type,
                 groups: list[list[int]], 
                 max_lag: int,
                 discovery_class: type[GroupCausalDiscovery],
                 aggregator: AggregationMap,
                 discovery_kwargs: dict[str, Any] | None = None,
                 p_val_threshold: float = 0.05, 
                 num_regimes: int = 1,
                 target_alpha_q: float = 0.8,
                 score_type: str = 'ac'):
        """
        Args:
            ci_test_class (type): Class of the conditional independence test to use.
            groups (List[List[int]]): List of groups, where each group is a list of variable indices.
            max_lag (int): Maximum lag to consider for time series data.
            discovery_class (type[GroupCausalDiscovery]): Class of the causal discovery model.
            aggregator (Any): Aggregation map instance (e.g., TunableDeepLatent or TunablePCA).
            discovery_kwargs (Optional[Dict]): Extra keyword arguments to pass to the discovery model on instantiation.
            p_val_threshold (float): Significance level for independence tests.
            num_regimes (int): Number of regimes for regime-switching models.
            target_alpha_q (float): Target score threshold for stopping the adaptive aggregation search.
            score_type (str): Type of score to evaluate ('c_ind', 'c_dep', or 'ac').
        """
        
        self.ci_test = ci_test_class
        self._groups = groups
        self.max_lag = max_lag
        self.alpha = p_val_threshold
        self.num_regimes = num_regimes
        
        self.discovery_class = discovery_class
        self.aggregator = aggregator
        self.target_alpha_q = target_alpha_q
        self.score_type = score_type
        self.discovery_kwargs = discovery_kwargs or {}
        
        valid_scores = ['c_ind', 'c_dep', 'ac']
        if self.score_type not in valid_scores:
            raise ValueError(f"score_type must be one of {valid_scores}")
            
        self._raw_group_data = None  
        self._cached_Zm = None

    def fit(self, X: list[torch.Tensor], U: list[torch.Tensor] | None = None, **kwargs) -> 'AdagWrapper':
        """Fits the aggregator to find optimal dimensions. Use fit_transform to get latents directly."""
        self.fit_transform(X, U, **kwargs)
        return self

    def transform(self, X: list[torch.Tensor], U: list[torch.Tensor] | None = None, **kwargs) -> list[torch.Tensor]:
        """Returns the discovered latent representations."""
        if self._cached_Zm is None:
            raise RuntimeError("AdagWrapper must be fitted before calling transform().")
        return self._cached_Zm

    def is_independent(self, p_val: float) -> bool:
        return p_val > self.alpha

    def fit_transform(self, X: list[torch.Tensor], U: list[torch.Tensor] | None = None, **kwargs) -> tuple[list[torch.Tensor], float, list[int]]:
        """
        Runs the Adag dimensionality search and transforms the data.
        Returns the latent representations, the achieved score, and the final dimensions.
        """
        self._raw_group_data = X
        
        N = len(X)
        m = [1] * N
        max_m = [x.shape[1] if x.ndim > 1 else 1 for x in X]
        
        current_score = 0.0
        Z_m = []
    
        while current_score < self.target_alpha_q:
            logging.debug(f"--- Adag Iteration | Current dimensions m: {m} ---")
            
            Z_m = []
            for i in range(N):
                U_i = U[i] if U is not None else None
                Z_m.append(self.aggregator.aggregate(X[i], m[i], U=U_i))
            
            # --- Integration with GroupCausalDiscovery ---
            # 1. Convert the list of tensors Z_m to a single numpy array for the discovery model
            Z_np = [z.detach().cpu().numpy() for z in Z_m]
            data_np = np.concatenate(Z_np, axis=1)

            # 2. Reconstruct the groups based on current dimensions m
            current_groups = []
            start_idx = 0
            for m_i in m:
                end_idx = start_idx + m_i
                current_groups.append(set(range(start_idx, end_idx)))
                start_idx = end_idx

            # 3. Instantiate a new discovery model with the current data and groups
            current_discovery = self.discovery_class(
                data=data_np, 
                groups=current_groups, 
                **self.discovery_kwargs
            )
            
            # 4. Extract the parents using the current discovery model
            group_parents = current_discovery.extract_parents()
            # --------------------------------------------

            logging.debug(f"Discovered independencies at m={m}: {group_parents}")
            
            # Compute respective aggregation consistency scores
            c_ind = self._compute_c_ind(group_parents)
            c_dep = self._compute_c_dep(group_parents)
            
            if self.score_type == 'c_ind':
                current_score = c_ind
            elif self.score_type == 'c_dep':
                current_score = c_dep
            elif self.score_type == 'ac':
                current_score = (c_ind + c_dep) / 2.0
                
            logging.debug(f"Target {self.score_type.upper()}: {self.target_alpha_q} | Achieved: {current_score:.3f}")
            logging.debug(f"[Details] c_ind: {c_ind:.3f} | c_dep: {c_dep:.3f}")
            
            if current_score >= self.target_alpha_q or m == max_m:
                break
                
            # Advance dimensions element-wise up to max_m 
            for i in range(N):
                if m[i] < max_m[i]:
                    m[i] += 1
                    
        return Z_m, current_score, m

    def _compute_c_ind(self, group_parents: dict[int, list[tuple[int, int]]]) -> float:
        """
        Evaluates independence consistency (c_ind) on the raw un-aggregated data.
        """
        if self._raw_group_data is None:
            raise RuntimeError("Raw group data is missing. AdagWrapper.run() must be called first.")
            
        C_ind_count = 0
        I_ind_count = 0
        N = len(self._groups)
        T = self._raw_group_data[0].shape[0]
        
        for j in range(N):
            parents_of_j = group_parents.get(j, [])
            max_z_lag = max([abs(lag) for _, lag in parents_of_j]) if parents_of_j else 0
            
            for i in range(N):
                for tau in range(0, self.max_lag + 1):
                    if i == j and tau == 0:
                        continue  # Skip self-dependency at lag 0
                    # Test non-adjacencies (conditional independencies)
                    if (i, -tau) not in parents_of_j:
                        
                        safe_max_lag = max(tau, max_z_lag)
                        start_t = safe_max_lag
                        end_t = T
                        
                        if start_t >= end_t - 5:
                            pval = 1.0 
                        else:
                            X_full = self._raw_group_data[i][start_t - tau : end_t - tau].to(torch.float32)
                            Y_full = self._raw_group_data[j][start_t : end_t].to(torch.float32)
                            
                            Z_full = None
                            if parents_of_j:
                                Z_list = [
                                    self._raw_group_data[z_var][start_t - abs(z_lag) : end_t - abs(z_lag)].to(torch.float32) 
                                    for z_var, z_lag in parents_of_j
                                ]
                                Z_full = torch.cat(Z_list, dim=1)
                            
                            if Z_full is not None:
                                logging.debug(f"Testing observed independence between group {i} (lag {tau}) and group {j} conditioned on parents of {j}.")
                                _, pval = self.ci_test.conditional_test(X_full, Y_full, Z_full)
                            else:
                                logging.debug(f"Testing observed independence between group {i} (lag {tau}) and group {j} with no conditioning set.")
                                _, pval = self.ci_test.test(X_full, Y_full)
                                
                            pval = max(float(pval), 1e-15)
                            
                        if self.is_independent(pval):
                            C_ind_count += 1
                        else:
                            I_ind_count += 1
                            
        total = C_ind_count + I_ind_count
        return C_ind_count / total if total > 0 else 1.0

    def _compute_c_dep(self, group_parents: dict[int, list[tuple[int, int]]]) -> float:
        """
        Evaluates effective dependence consistency (c_dep_bar) on the raw un-aggregated data.
        Checks if aggregate adjacencies map to true vector-level dependencies.
        """
        if self._raw_group_data is None:
            raise RuntimeError("Raw group data is missing. AdagWrapper.run() must be called first.")
            
        C_adj_count = 0
        I_adj_count = 0
        N = len(self._groups)
        T = self._raw_group_data[0].shape[0]

        for j in range(N):
            parents_of_j = group_parents.get(j, [])
            
            for (p_var, p_lag) in parents_of_j:
                # The condition set is all other aggregate parents of j except the one currently being tested
                cond_set = [p for p in parents_of_j if p != (p_var, p_lag)]
                max_z_lag = max([abs(lag) for _, lag in cond_set]) if cond_set else 0
                
                safe_max_lag = max(abs(p_lag), max_z_lag)
                start_t = safe_max_lag
                end_t = T
                
                if start_t >= end_t - 5:
                    pval = 0.0 # Assume dependent to avoid unfairly penalizing with sparse data
                else:
                    X_full = self._raw_group_data[p_var][start_t - abs(p_lag) : end_t - abs(p_lag)].to(torch.float32)
                    Y_full = self._raw_group_data[j][start_t : end_t].to(torch.float32)
                    
                    Z_full = None
                    if cond_set:
                        Z_list = [
                            self._raw_group_data[z_var][start_t - abs(z_lag) : end_t - abs(z_lag)].to(torch.float32) 
                            for z_var, z_lag in cond_set
                        ]
                        Z_full = torch.cat(Z_list, dim=1)
                    
                    if Z_full is not None:
                        _, pval = self.ci_test.conditional_test(X_full, Y_full, Z_full)
                    else:
                        _, pval = self.ci_test.test(X_full, Y_full)
                        
                    pval = max(float(pval), 1e-15)
                
                # We expect dependence (pval <= alpha). If it is dependent, it's consistent.
                if not self.is_independent(pval):
                    C_adj_count += 1
                else:
                    I_adj_count += 1
                    
        total = C_adj_count + I_adj_count
        return C_adj_count / total if total > 0 else 1.0