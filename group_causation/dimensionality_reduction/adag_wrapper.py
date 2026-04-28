import torch
from typing import Callable, List, Tuple, Optional

from group_causation.dimensionality_reduction.iVAE.wrappers import IVAEDimensionalityReduction, VAEDimensionalityReduction

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
    


class AdagWrapper:
    """
    Adaptive Aggregation (Adag) wrapper for Causal Discovery over vector-valued variables.
    """
    def __init__(self, ci_test_class, p_val_threshold: float = 0.05):
        self.ci_test = ci_test_class
        self.alpha = p_val_threshold

    def is_independent(self, p_val: float) -> bool:
        return p_val > self.alpha

    def compute_c_ind(self, 
                      X_full: List[torch.Tensor], 
                      Z_independencies: List[Tuple[int, int, List[int]]], 
                      **kwargs) -> float:
        """Computes the independence score for aggregation consistency (c_ind)."""
        if not Z_independencies:
            return 1.0 

        C_ind_count = 0  
        I_ind_count = 0  

        for i, j, cond_idxs in Z_independencies:
            X_i = X_full[i]
            X_j = X_full[j]
            
            if cond_idxs:
                X_cond = torch.cat([X_full[k] for k in cond_idxs], dim=1)
                _, p_val = self.ci_test.conditional_test(X_i, X_j, X_cond, **kwargs)
            else:
                _, p_val = self.ci_test.test(X_i, X_j, **kwargs)
                
            if self.is_independent(p_val):
                C_ind_count += 1
            else:
                I_ind_count += 1

        total = C_ind_count + I_ind_count
        return C_ind_count / total if total > 0 else 1.0

    def run(self, 
            X_data: List[torch.Tensor], 
            discovery_func: Callable, 
            aggregator: TunableDeepLatent,
            U_data: Optional[List[torch.Tensor]] = None,
            target_alpha_q: float = 0.8,
            **kwargs) -> Tuple[List[torch.Tensor], float, List[int]]:
        """
        Executes the Adag Loop with a dynamic aggregator.
        
        Args:
            X_data: List of original high-dimensional tensor variables [X^1, X^2, ..., X^N].
            discovery_func: Callable Causal Discovery algorithm (e.g., PC skeleton phase).
            aggregator: Instantiated TunableDeepLatent (or TunablePCA) object.
            U_data: Optional list of auxiliary variables [U^1, U^2, ..., U^N] for iVAE.
            target_alpha_q: The desired consistency score (c_ind).
            **kwargs: Extra parameters passed to the underlying full-dimensional CI tests.
        """
        N = len(X_data)
        m = [1] * N
        max_m = [x.shape[1] if x.ndim > 1 else 1 for x in X_data]
        
        c_ind_score = 0.0
        Z_m = []

        while c_ind_score < target_alpha_q:
            print(f"--- Adag Iteration | Current dimensions m: {m} ---")
            
            # 1. Define aggregate variables Z_m = g^m(X) using VAE/iVAE
            Z_m = []
            for i in range(N):
                U_i = U_data[i] if U_data is not None else None
                Z_m.append(aggregator.aggregate(X_data[i], m[i], U=U_i))
            
            # 2. Run CD algorithm over Z_m to get the independence model I(Z)
            Z_independencies = discovery_func(Z_m)
            
            # 3. Compute independence consistency score (c_ind) by verifying on full X
            c_ind_score = self.compute_c_ind(X_data, Z_independencies, **kwargs)
            
            print(f"Target c_ind: {target_alpha_q} | Achieved c_ind: {c_ind_score:.3f}")
            
            # 4. Check termination condition
            if c_ind_score >= target_alpha_q or m == max_m:
                break
                
            # 5. Update step: m += [1]^N
            for i in range(N):
                if m[i] < max_m[i]:
                    m[i] += 1
                    
        return Z_m, c_ind_score, m