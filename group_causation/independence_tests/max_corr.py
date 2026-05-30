import torch
import math
from scipy.stats import t

from group_causation.independence_tests.conditional_independence_base import ConditionalIndependence_base

class MaxCorr_Test(ConditionalIndependence_base):
    """Max-Corr Conditional Independence Test using Bonferroni-corrected Pearson correlations (PyTorch Accelerated)."""
    
    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def test(cls, X: torch.Tensor, Y: torch.Tensor, max_samples=500, n_ensembles=5, sequential_chunks=False) -> tuple[float, float]:
        X = X.view(-1, 1) if X.ndim == 1 else X
        Y = Y.view(-1, 1) if Y.ndim == 1 else Y
        n = X.shape[0]
        
        if n <= max_samples:
            return cls._single_test(X, Y)
            
        p_vals = []
        stats = []
        weights = []
        
        if sequential_chunks:
            num_chunks = max(1, math.ceil(n / max_samples))
            chunks_X = torch.tensor_split(X, num_chunks)
            chunks_Y = torch.tensor_split(Y, num_chunks)
            
            valid_samples = sum(cX.shape[0] for cX in chunks_X if cX.shape[0] >= 6)
            if valid_samples == 0: 
                return 0.0, 1.0

            for cX, cY in zip(chunks_X, chunks_Y):
                n_local = cX.shape[0]
                if n_local < 6:
                    continue
                s, p = cls._single_test(cX, cY)
                p = max(1e-15, min(1.0 - 1e-15, p))
                
                stats.append(s)
                p_vals.append(p)
                weights.append(n_local / valid_samples) # Uniform weights proportional to chunk size
                
            if not p_vals:
                return 0.0, 1.0
        else:
            for _ in range(n_ensembles):
                idx = torch.randperm(n, device=X.device)[:max_samples]
                s, p = cls._single_test(X[idx], Y[idx])
                p = max(1e-15, min(1.0 - 1e-15, p))
                
                stats.append(s)
                p_vals.append(p)
                weights.append(1.0 / n_ensembles) # Uniform weights for random subsampling
            
        return cls._aggregate_cct(stats, p_vals, weights)

    @classmethod
    def conditional_test(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor,
                         max_samples=500, n_ensembles=5, sequential_chunks=False,
                         ridge_lambda: float = 0.2) -> tuple[float, float]:
        '''
        Conditional Max-Corr test with optional subsampling and sequential chunking for large datasets.
        
        Args:
            X (torch.Tensor): Source variable tensor.
            Y (torch.Tensor): Target variable tensor.
            Z (torch.Tensor): Conditioning set tensor.
            max_samples (int): Maximum samples for each test when subsampling.
            n_ensembles (int): Number of random subsamples to average over when not using sequential chunks.
            sequential_chunks (bool): Whether to use sequential non-overlapping chunks instead of random subsampling.
            ridge_lambda (float): Regularization strength for ridge regression in the conditional test.
        Returns:
            tuple[float, float]: (average_statistic, global_p_value) where average_statistic is the weighted average of local Max-Corr statistics and global_p_value is the aggregated p-value from the Cauchy Combination Test.

        '''
        X = X.view(-1, 1) if X.ndim == 1 else X
        Y = Y.view(-1, 1) if Y.ndim == 1 else Y
        Z = Z.view(-1, 1) if Z.ndim == 1 else Z
        n = X.shape[0]
        
        if n <= max_samples:
            return cls._single_conditional_test(X, Y, Z, ridge_lambda=ridge_lambda)
            
        p_vals = []
        stats = []
        weights = []
        
        if sequential_chunks:
            num_chunks = max(1, math.ceil(n / max_samples))
            chunks_X = torch.tensor_split(X, num_chunks)
            chunks_Y = torch.tensor_split(Y, num_chunks)
            chunks_Z = torch.tensor_split(Z, num_chunks)
            
            valid_samples = sum(cX.shape[0] for cX in chunks_X if cX.shape[0] >= 6)
            if valid_samples == 0: 
                return 0.0, 1.0

            for cX, cY, cZ in zip(chunks_X, chunks_Y, chunks_Z):
                n_local = cX.shape[0]
                if n_local < 6: 
                    continue
                s, p = cls._single_conditional_test(cX, cY, cZ, ridge_lambda=ridge_lambda)
                p = max(1e-15, min(1.0 - 1e-15, p))
                
                stats.append(s)
                p_vals.append(p)
                weights.append(n_local / valid_samples)
                
            if not p_vals: 
                return 0.0, 1.0
        else:
            for _ in range(n_ensembles):
                idx = torch.randperm(n, device=X.device)[:max_samples]
                s, p = cls._single_conditional_test(X[idx], Y[idx], Z[idx], ridge_lambda=ridge_lambda)
                p = max(1e-15, min(1.0 - 1e-15, p))
                
                stats.append(s)
                p_vals.append(p)
                weights.append(1.0 / n_ensembles) # Uniform weights for random subsampling
            
        return cls._aggregate_cct(stats, p_vals, weights)

    @classmethod
    def test_regimes(cls, X_regimes: list[torch.Tensor], Y_regimes: list[torch.Tensor]) -> tuple[float, float]:
        """
        Unconditional test across regimes using the Cauchy Combination Test (CCT).
        """
        if not X_regimes:
            return 0.0, 1.0

        p_vals = []
        stats = []
        weights = []
        
        valid_samples = sum(X.shape[0] for X in X_regimes if X.shape[0] >= 6)

        if valid_samples == 0:
            return 0.0, 1.0

        for X_local, Y_local in zip(X_regimes, Y_regimes):
            n_local = X_local.shape[0]
            if n_local < 6:
                continue
                
            s, p = cls._single_test(X_local, Y_local)
            p = max(1e-15, min(1.0 - 1e-15, p))
            
            stats.append(s)
            p_vals.append(p)
            weights.append(n_local / valid_samples)

        if not p_vals:
            return 0.0, 1.0

        return cls._aggregate_cct(stats, p_vals, weights)

    @classmethod
    def conditional_test_regimes(cls, X_regimes: list[torch.Tensor], Y_regimes: list[torch.Tensor], Z_regimes: list[torch.Tensor], ridge_lambda: float = 0.2) -> tuple[float, float]:
        """
        Conditional test across regimes using the Cauchy Combination Test (CCT).

        Evaluates the conditional independence X ⫫ Y | Z across multiple non-stationary 
        regimes using the Cauchy Combination Test (CCT).

        This method computes local Max-Corr p-values per regime and aggregates them using CCT. 
        This guarantees theoretical robustness against cross-regime temporal dependencies 
        and covariance cancellation, without requiring empirical estimation of the 
        correlation matrix between statistical tests.

        Args:
            X_regimes (list[torch.Tensor]): List of tensors for the source variable X, 
                where each tensor represents a temporally continuous regime.
            Y_regimes (list[torch.Tensor]): List of tensors for the target variable Y.
            Z_regimes (list[torch.Tensor]): List of tensors for the conditioning set Z.

        Returns:
            tuple[float, float]:
                - avg_stat: Weighted average of the local Max-Corr statistics (Proxy effect size).
                - global_p_val: The exact analytical p-value derived from the standard 
                  Cauchy Cumulative Distribution Function (CDF).

        Notes:
            - Regimes with fewer than 6 samples are excluded from the test to ensure 
              minimum numerical stability in Ordinary Least Squares (OLS) regression.
            - Degrees of freedom are dynamically adjusted per regime as `max(1, n - 2 - dz)` 
              to account for the parameters consumed by the local conditioning set Z.
        """
        if not X_regimes:
            return 0.0, 1.0

        p_vals = []
        stats = []
        weights = []
        
        valid_samples = sum(X.shape[0] for X in X_regimes if X.shape[0] >= 6)

        if valid_samples == 0:
            return 0.0, 1.0

        for X_local, Y_local, Z_local in zip(X_regimes, Y_regimes, Z_regimes):
            n_local = X_local.shape[0]
            if n_local < 6:
                continue

            s, p = cls._single_conditional_test(X_local, Y_local, Z_local, ridge_lambda=ridge_lambda)
            p = max(1e-15, min(1.0 - 1e-15, p))
            
            stats.append(s)
            p_vals.append(p)
            weights.append(n_local / valid_samples)

        if not p_vals:
            return 0.0, 1.0

        return cls._aggregate_cct(stats, p_vals, weights)

    @classmethod
    def _single_conditional_test(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, ridge_lambda: float) -> tuple[float, float]:
        n = X.shape[0]
        if n < 6:
            return 0.0, 1.0 

        Z_int = torch.cat([Z, torch.ones(n, 1, dtype=Z.dtype, device=Z.device)], dim=1)
        
        I = torch.eye(Z_int.shape[1], device=Z_int.device, dtype=Z_int.dtype)
        
        ZtZ_ridge = Z_int.T @ Z_int + ridge_lambda * I
        
        beta_X = torch.linalg.solve(ZtZ_ridge, Z_int.T @ X)
        beta_Y = torch.linalg.solve(ZtZ_ridge, Z_int.T @ Y)
        
        rX = X - Z_int @ beta_X
        rY = Y - Z_int @ beta_Y
        
        # Calculus of effective degrees of freedom (Trace of the Hat Matrix)
        # tr( Z * (Z^T Z + \lambda I)^-1 * Z^T ) is mathematically identical to tr( (Z^T Z + \lambda I)^-1 * Z^T Z )
        # This second form is computationally much cheaper because it operates in dimension d_z x d_z instead of n x n
        ZtZ = Z_int.T @ Z_int
        hat_trace = torch.trace(torch.linalg.solve(ZtZ_ridge, ZtZ)).item()
        
        return cls._compute_max_corr_pval(rX, rY, degrees_of_freedom_consumed=hat_trace)

    @classmethod
    def _compute_max_corr_pval(cls, X: torch.Tensor, Y: torch.Tensor,
                               degrees_of_freedom_consumed: float = 0.0) -> tuple[float, float]:
        n = X.shape[0]
        dimX = X.shape[1]
        dimY = Y.shape[1]
        
        X_c = X - X.mean(dim=0, keepdim=True)
        Y_c = Y - Y.mean(dim=0, keepdim=True)
        
        X_norm = X_c / torch.clamp(torch.linalg.vector_norm(X_c, dim=0, keepdim=True), min=1e-8)
        Y_norm = Y_c / torch.clamp(torch.linalg.vector_norm(Y_c, dim=0, keepdim=True), min=1e-8)
        
        corr_matrix = X_norm.T @ Y_norm
        max_corr = torch.max(torch.abs(corr_matrix)).item()
        
        r = min(max_corr, 1.0 - 1e-8) 
        
        df = max(1.0, float(n - 2) - degrees_of_freedom_consumed)
        t_stat = r * math.sqrt(df / (1.0 - r**2))
        
        p_val_single = 2 * t.sf(t_stat, df=df) 
        
        num_tests = dimX * dimY
        p_val_bonf = min(1.0, p_val_single * num_tests)
        
        return float(max_corr), float(p_val_bonf)

    @staticmethod
    def _aggregate_cct(stats: list[float], p_vals: list[float], weights: list[float]) -> tuple[float, float]:
        """
        Aggregates p-values using the Cauchy Combination Test.
        """
        t_stat = 0.0
        for w, p in zip(weights, p_vals):
            t_stat += w * math.tan(math.pi * (0.5 - p))
            
        global_p_val = 0.5 - (math.atan(t_stat) / math.pi)
        avg_stat = sum(w * s for w, s in zip(weights, stats))
        
        return avg_stat, global_p_val

    @classmethod
    def _single_test(cls, X: torch.Tensor, Y: torch.Tensor) -> tuple[float, float]:
        n = X.shape[0]
        if n < 6:
            return 0.0, 1.0 
            
        return cls._compute_max_corr_pval(X, Y)