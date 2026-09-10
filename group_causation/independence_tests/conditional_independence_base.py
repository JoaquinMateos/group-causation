"""Base class for Conditional Independence Tests (PyTorch Accelerated)."""

import math
import statistics
from abc import ABC, abstractmethod

import torch


def _get_device() -> torch.device:
    """Return the best available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ConditionalIndependence_base(ABC):
    """Abstract base for conditional independence tests.

    Subclasses implement two hooks:

    * ``_single_test(X, Y, **kwargs)``       — unconditional test on a chunk
    * ``_single_conditional_test(X, Y, Z, **kwargs)`` — conditional test on a chunk

    The public ``test`` / ``conditional_test`` class methods handle
    dimensional consistency, chunking, ensembling, and aggregation.
    Override ``_aggregate_results`` to change the aggregation strategy
    (the default uses the median p-value and arithmetic mean of stats).
    """

    # ------------------------------------------------------------------
    # Hooks — subclasses must implement these
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def _single_test(cls, X: torch.Tensor, Y: torch.Tensor, **kwargs) -> tuple[float, float]:
        """Compute the unconditional test statistic and p-value for one chunk."""

    @classmethod
    @abstractmethod
    def _single_conditional_test(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, **kwargs) -> tuple[float, float]:
        """Compute the conditional test statistic and p-value for one chunk."""

    # ------------------------------------------------------------------
    # Aggregation — override to change how chunk results are combined
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_pval(p: float) -> float:
        """Clamp a p-value to avoid numerical issues in aggregation."""
        return max(1e-15, min(1.0 - 1e-15, p))

    @classmethod
    def _compute_weights(cls, n_chunks: int, chunk_sizes: list[int],
                         sequential_chunks: bool, **kwargs) -> list[float]:
        """Compute per-chunk weights for aggregation.

        Default: uniform weights.  Override for weighted schemes (e.g. CCT).
        """
        return [1.0 / n_chunks] * n_chunks

    @staticmethod
    def _aggregate_results(stats: list[float], p_vals: list[float],
                           weights: list[float] | None = None, **kwargs) -> tuple[float, float]:
        """Combine per-chunk results into a single (statistic, p_value).

        Default: arithmetic mean of statistics, median of p-values.
        Subclasses that need weighted aggregation (e.g. CCT) should
        override this and accept ``weights`` via ``**kwargs``.
        """
        if weights is None:
            weights = [1.0 / len(p_vals)] * len(p_vals)
        return sum(s * w for s, w in zip(stats, weights)), statistics.median(p_vals)

    # ------------------------------------------------------------------
    # Template methods — callers use these
    # ------------------------------------------------------------------

    @classmethod
    def test(cls, X: torch.Tensor, Y: torch.Tensor, max_samples: int = 500,
             n_ensembles: int = 5, sequential_chunks: bool = False,
             **kwargs) -> tuple[float, float]:
        """Unconditional independence test with automatic chunking / ensembling."""
        X = X.view(-1, 1) if X.ndim == 1 else X
        Y = Y.view(-1, 1) if Y.ndim == 1 else Y
        n = X.shape[0]

        if n <= max_samples:
            return cls._single_test(X, Y, **kwargs)

        p_vals: list[float] = []
        stats: list[float] = []
        chunk_sizes: list[int] = []

        if sequential_chunks:
            num_chunks = max(1, math.ceil(n / max_samples))
            chunks_X = torch.tensor_split(X, num_chunks)
            chunks_Y = torch.tensor_split(Y, num_chunks)

            for cX, cY in zip(chunks_X, chunks_Y):
                if cX.shape[0] < 6:
                    continue
                s, p = cls._single_test(cX, cY, **kwargs)
                p_vals.append(cls._clamp_pval(p))
                stats.append(s)
                chunk_sizes.append(cX.shape[0])

            if not p_vals:
                return 0.0, 1.0
        else:
            for _ in range(n_ensembles):
                idx = torch.randperm(n, device=X.device)[:max_samples]
                s, p = cls._single_test(X[idx], Y[idx], **kwargs)
                p_vals.append(cls._clamp_pval(p))
                stats.append(s)
                chunk_sizes.append(max_samples)

        weights = cls._compute_weights(len(p_vals), chunk_sizes, sequential_chunks, **kwargs)
        return cls._aggregate_results(stats, p_vals, weights=weights, **kwargs)

    @classmethod
    def conditional_test(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor,
                         max_samples: int = 500, n_ensembles: int = 5,
                         sequential_chunks: bool = False,
                         **kwargs) -> tuple[float, float]:
        """Conditional independence test with automatic chunking / ensembling."""
        X = X.view(-1, 1) if X.ndim == 1 else X
        Y = Y.view(-1, 1) if Y.ndim == 1 else Y
        Z = Z.view(-1, 1) if Z.ndim == 1 else Z
        n = X.shape[0]

        if n <= max_samples:
            return cls._single_conditional_test(X, Y, Z, **kwargs)

        p_vals: list[float] = []
        stats: list[float] = []
        chunk_sizes: list[int] = []

        if sequential_chunks:
            num_chunks = max(1, math.ceil(n / max_samples))
            chunks_X = torch.tensor_split(X, num_chunks)
            chunks_Y = torch.tensor_split(Y, num_chunks)
            chunks_Z = torch.tensor_split(Z, num_chunks)

            for cX, cY, cZ in zip(chunks_X, chunks_Y, chunks_Z):
                if cX.shape[0] < 6:
                    continue
                s, p = cls._single_conditional_test(cX, cY, cZ, **kwargs)
                p_vals.append(cls._clamp_pval(p))
                stats.append(s)
                chunk_sizes.append(cX.shape[0])

            if not p_vals:
                return 0.0, 1.0
        else:
            for _ in range(n_ensembles):
                idx = torch.randperm(n, device=X.device)[:max_samples]
                s, p = cls._single_conditional_test(X[idx], Y[idx], Z[idx], **kwargs)
                p_vals.append(cls._clamp_pval(p))
                stats.append(s)
                chunk_sizes.append(max_samples)

        weights = cls._compute_weights(len(p_vals), chunk_sizes, sequential_chunks, **kwargs)
        return cls._aggregate_results(stats, p_vals, weights=weights, **kwargs)

    # ------------------------------------------------------------------
    # Regime methods — optional overrides
    # ------------------------------------------------------------------

    @classmethod
    def test_regimes(cls, X_regimes: list[torch.Tensor], Y_regimes: list[torch.Tensor], **kwargs) -> tuple[float, float]:
        """Override in subclasses that support multi-regime unconditional testing."""
        raise NotImplementedError(f"{cls.__name__} does not support multi-regime testing.")

    @classmethod
    def conditional_test_regimes(cls, X_regimes: list[torch.Tensor], Y_regimes: list[torch.Tensor],
                                 Z_regimes: list[torch.Tensor], **kwargs) -> tuple[float, float]:
        """Override in subclasses that support multi-regime conditional testing."""
        raise NotImplementedError(f"{cls.__name__} does not support multi-regime conditional testing.")
