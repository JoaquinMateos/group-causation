import torch
import math
import statistics
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
        
        if sequential_chunks:
            num_chunks = max(1, math.ceil(n / max_samples))
            chunks_X = torch.tensor_split(X, num_chunks)
            chunks_Y = torch.tensor_split(Y, num_chunks)
            
            for cX, cY in zip(chunks_X, chunks_Y):
                if cX.shape[0] < 6:
                    continue
                s, p = cls._single_test(cX, cY)
                p_vals.append(p)
                stats.append(s)
                
            if not p_vals:
                return 0.0, 1.0
        else:
            for _ in range(n_ensembles):
                idx = torch.randperm(n, device=X.device)[:max_samples]
                s, p = cls._single_test(X[idx], Y[idx])
                p_vals.append(p)
                stats.append(s)
            
        return sum(stats) / len(stats), statistics.median(p_vals)

    @classmethod
    def conditional_test(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, max_samples=500, n_ensembles=5, sequential_chunks=False) -> tuple[float, float]:
        X = X.view(-1, 1) if X.ndim == 1 else X
        Y = Y.view(-1, 1) if Y.ndim == 1 else Y
        Z = Z.view(-1, 1) if Z.ndim == 1 else Z
        n = X.shape[0]
        
        if n <= max_samples:
            return cls._single_conditional_test(X, Y, Z)
            
        p_vals = []
        stats = []
        
        if sequential_chunks:
            num_chunks = max(1, math.ceil(n / max_samples))
            chunks_X = torch.tensor_split(X, num_chunks)
            chunks_Y = torch.tensor_split(Y, num_chunks)
            chunks_Z = torch.tensor_split(Z, num_chunks)
            
            for cX, cY, cZ in zip(chunks_X, chunks_Y, chunks_Z):
                if cX.shape[0] < 6: 
                    continue
                s, p = cls._single_conditional_test(cX, cY, cZ)
                p_vals.append(p)
                stats.append(s)
                
            if not p_vals: 
                return 0.0, 1.0
        else:
            for _ in range(n_ensembles):
                idx = torch.randperm(n, device=X.device)[:max_samples]
                s, p = cls._single_conditional_test(X[idx], Y[idx], Z[idx])
                p_vals.append(p)
                stats.append(s)
            
        return sum(stats) / len(stats), statistics.median(p_vals)

    @classmethod
    def test_regimes(cls, X_regimes: list[torch.Tensor], Y_regimes: list[torch.Tensor]) -> tuple[float, float]:
        """
        Performs an unconditional Max-Corr test across multiple regimes using Pooled Standardized Residuals.
        """
        pooled_rX = []
        pooled_rY = []

        for X_local, Y_local in zip(X_regimes, Y_regimes):
            # For unconditional tests, the "residual" is just the mean-centered variable
            rX = X_local - X_local.mean(dim=0, keepdim=True)
            rY = Y_local - Y_local.mean(dim=0, keepdim=True)

            # Local Z-score standardization (0 mean, 1 variance) to remove heteroscedasticity across regimes
            std_X = torch.clamp(rX.std(dim=0, unbiased=True, keepdim=True), min=1e-8)
            std_Y = torch.clamp(rY.std(dim=0, unbiased=True, keepdim=True), min=1e-8)

            pooled_rX.append(rX / std_X)
            pooled_rY.append(rY / std_Y)

        if not pooled_rX:
            return 0.0, 1.0

        # Early Fusion: Concatenate all standardized residuals into a single global distribution
        rX_concat = torch.cat(pooled_rX, dim=0)
        rY_concat = torch.cat(pooled_rY, dim=0)

        # Execute a single, high-powered Max-Corr test
        return cls._compute_max_corr_pval(rX_concat, rY_concat)

    @classmethod
    def conditional_test_regimes(cls, X_regimes: list[torch.Tensor], Y_regimes: list[torch.Tensor], Z_regimes: list[torch.Tensor]) -> tuple[float, float]:
        """
        Performs a conditional Max-Corr test across multiple regimes using Pooled Standardized Residuals.
        """
        pooled_rX = []
        pooled_rY = []

        for X_local, Y_local, Z_local in zip(X_regimes, Y_regimes, Z_regimes):
            n = X_local.shape[0]

            # 1. Local Ordinary Least Squares (OLS) Regression
            # Add an intercept term to Z_local to absorb the local mean of the regime
            Z_int = torch.cat([Z_local, torch.ones(n, 1, dtype=Z_local.dtype, device=Z_local.device)], dim=1)
            
            beta_X = torch.linalg.lstsq(Z_int, X_local).solution
            beta_Y = torch.linalg.lstsq(Z_int, Y_local).solution
            
            # Extract local residuals
            rX = X_local - Z_int @ beta_X
            rY = Y_local - Z_int @ beta_Y

            # 2. Local Z-score standardization (0 mean, 1 variance)
            std_X = torch.clamp(rX.std(dim=0, unbiased=True, keepdim=True), min=1e-8)
            std_Y = torch.clamp(rY.std(dim=0, unbiased=True, keepdim=True), min=1e-8)

            pooled_rX.append(rX / std_X)
            pooled_rY.append(rY / std_Y)

        if not pooled_rX:
            return 0.0, 1.0

        # 3. Early Fusion: Concatenate all standardized residuals into a single global distribution
        rX_concat = torch.cat(pooled_rX, dim=0)
        rY_concat = torch.cat(pooled_rY, dim=0)

        # 4. Execute a single, high-powered Max-Corr test on the pooled data
        return cls._compute_max_corr_pval(rX_concat, rY_concat)

    @classmethod
    def _single_test(cls, X: torch.Tensor, Y: torch.Tensor) -> tuple[float, float]:
        n = X.shape[0]
        if n < 6:
            return 0.0, 1.0 
            
        return cls._compute_max_corr_pval(X, Y)

    @classmethod
    def _single_conditional_test(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor) -> tuple[float, float]:
        n = X.shape[0]
        if n < 6:
            return 0.0, 1.0 

        # 1. Regress X on Z and Y on Z to get residuals
        # Add an intercept term to Z to ensure residuals have zero mean
        Z_int = torch.cat([Z, torch.ones(n, 1, dtype=Z.dtype, device=Z.device)], dim=1)
        
        # Solve Ordinary Least Squares
        beta_X = torch.linalg.lstsq(Z_int, X).solution
        beta_Y = torch.linalg.lstsq(Z_int, Y).solution
        
        # Calculate residuals (rX_Z, rY_Z)
        rX = X - Z_int @ beta_X
        rY = Y - Z_int @ beta_Y
        
        # 2. Compute Max-Corr on the residuals
        return cls._compute_max_corr_pval(rX, rY)

    @classmethod
    def _compute_max_corr_pval(cls, X: torch.Tensor, Y: torch.Tensor) -> tuple[float, float]:
        n = X.shape[0]
        dimX = X.shape[1]
        dimY = Y.shape[1]
        
        # Mean center the variables
        X_c = X - X.mean(dim=0, keepdim=True)
        Y_c = Y - Y.mean(dim=0, keepdim=True)
        
        # L2 normalize the columns to prepare for Pearson correlation
        X_norm = X_c / torch.clamp(torch.linalg.vector_norm(X_c, dim=0, keepdim=True), min=1e-8)
        Y_norm = Y_c / torch.clamp(torch.linalg.vector_norm(Y_c, dim=0, keepdim=True), min=1e-8)
        
        # Compute pairwise Pearson correlation matrix
        corr_matrix = X_norm.T @ Y_norm
        
        # Get the test statistic: max absolute correlation (rho)
        max_corr = torch.max(torch.abs(corr_matrix)).item()
        
        # Compute the t-statistic for the maximum correlation
        r = min(max_corr, 1.0 - 1e-8)  # Clamp to prevent division by zero
        t_stat = r * math.sqrt((n - 2) / (1.0 - r**2))
        
        # Compute the base two-sided p-value from the Student's t-distribution
        p_val_single = 2 * t.sf(t_stat, df=n - 2)
        
        # Apply the Bonferroni correction across all tested pairs
        num_tests = dimX * dimY
        p_val_bonf = min(1.0, p_val_single * num_tests)
        
        return float(max_corr), float(p_val_bonf)