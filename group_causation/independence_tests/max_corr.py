"""Max-Corr Conditional Independence Test using Bonferroni-corrected Pearson correlations (PyTorch Accelerated)."""

import math

import torch
from scipy.stats import t

from group_causation.independence_tests.conditional_independence_base import (
    ConditionalIndependence_base,
    _get_device,
)


class MaxCorr_Test(ConditionalIndependence_base):
    """Max-Corr conditional independence test with CCT aggregation."""

    supports_regime_testing = True

    # ------------------------------------------------------------------
    # Chunk-weight computation — overrides the base default
    # ------------------------------------------------------------------

    @classmethod
    def _compute_weights(cls, n_chunks: int, chunk_sizes: list[int],
                         sequential_chunks: bool, **kwargs) -> list[float]:
        """Return per-chunk weights for CCT aggregation.

        Sequential chunks: weight proportional to chunk size.
        Random subsamples: uniform weights.
        """
        if sequential_chunks:
            total = sum(chunk_sizes)
            return [s / total for s in chunk_sizes]
        return [1.0 / n_chunks] * n_chunks

    # ------------------------------------------------------------------
    # Aggregation — weighted CCT instead of simple median
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_results(stats: list[float], p_vals: list[float],
                           weights: list[float] | None = None, **kwargs) -> tuple[float, float]:
        """Aggregate p-values using the Cauchy Combination Test (CCT)."""
        if weights is None:
            weights = [1.0 / len(p_vals)] * len(p_vals)

        t_stat = 0.0
        for w, p in zip(weights, p_vals):
            t_stat += w * math.tan(math.pi * (0.5 - p))

        global_p_val = 0.5 - (math.atan(t_stat) / math.pi)
        avg_stat = sum(w * s for w, s in zip(weights, stats))

        return avg_stat, global_p_val

    # ------------------------------------------------------------------
    # Required hooks from ConditionalIndependence_base
    # ------------------------------------------------------------------

    @classmethod
    def _single_test(cls, X: torch.Tensor, Y: torch.Tensor) -> tuple[float, float]:
        n = X.shape[0]
        if n < 6:
            return 0.0, 1.0
        return cls._compute_max_corr_pval(X, Y)

    @classmethod
    def _single_conditional_test(cls, X: torch.Tensor, Y: torch.Tensor,
                                 Z: torch.Tensor, ridge_lambda: float = 0.2) -> tuple[float, float]:
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

        ZtZ = Z_int.T @ Z_int
        hat_trace = torch.trace(torch.linalg.solve(ZtZ_ridge, ZtZ)).item()

        return cls._compute_max_corr_pval(rX, rY, degrees_of_freedom_consumed=hat_trace)

    # ------------------------------------------------------------------
    # Regime methods — CCT aggregation across regimes
    # ------------------------------------------------------------------

    @classmethod
    def test_regimes(cls, X_regimes: list[torch.Tensor], Y_regimes: list[torch.Tensor]) -> tuple[float, float]:
        """Unconditional test across regimes using CCT."""
        if not X_regimes:
            return 0.0, 1.0

        p_vals: list[float] = []
        stats: list[float] = []
        weights: list[float] = []

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

        return cls._aggregate_results(stats, p_vals, weights=weights)

    @classmethod
    def conditional_test_regimes(cls, X_regimes: list[torch.Tensor], Y_regimes: list[torch.Tensor],
                                 Z_regimes: list[torch.Tensor],
                                 ridge_lambda: float = 0.2) -> tuple[float, float]:
        """Conditional test across regimes using CCT."""
        if not X_regimes:
            return 0.0, 1.0

        p_vals: list[float] = []
        stats: list[float] = []
        weights: list[float] = []

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

        return cls._aggregate_results(stats, p_vals, weights=weights)

    # ------------------------------------------------------------------
    # Max-Corr internals
    # ------------------------------------------------------------------

    # Backward-compatible alias for the old method name
    _aggregate_cct = _aggregate_results

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
        t_stat = r * math.sqrt(df / (1.0 - r ** 2))

        p_val_single = 2 * t.sf(t_stat, df=df)

        num_tests = dimX * dimY
        p_val_bonf = min(1.0, p_val_single * num_tests)

        return float(max_corr), float(p_val_bonf)
