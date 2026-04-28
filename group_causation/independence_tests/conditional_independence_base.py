import torch
import math
import statistics
from abc import ABC, abstractmethod

class ConditionalIndependence_base(ABC):
    """Base class for Conditional Independence Tests (PyTorch Accelerated)."""

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    @abstractmethod
    def _single_test(cls, X: torch.Tensor, Y: torch.Tensor, **kwargs) -> tuple[float, float]:
        """To be implemented by subclasses: purely computes the unconditional test statistic and p-value."""
        pass

    @classmethod
    @abstractmethod
    def _single_conditional_test(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, **kwargs) -> tuple[float, float]:
        """To be implemented by subclasses: purely computes the conditional test statistic and p-value."""
        pass

    @classmethod
    def test(cls, X: torch.Tensor, Y: torch.Tensor, max_samples=500, n_ensembles=5, sequential_chunks=False, **kwargs) -> tuple[float, float]:
        """Base test handler to manage dimensional consistency, chunking, and ensembling."""
        X = X.view(-1, 1) if X.ndim == 1 else X
        Y = Y.view(-1, 1) if Y.ndim == 1 else Y
        n = X.shape[0]
        
        if n <= max_samples:
            return cls._single_test(X, Y, **kwargs)
            
        p_vals = []
        stats = []
        
        if sequential_chunks:
            num_chunks = max(1, math.ceil(n / max_samples))
            chunks_X = torch.tensor_split(X, num_chunks)
            chunks_Y = torch.tensor_split(Y, num_chunks)
            
            for cX, cY in zip(chunks_X, chunks_Y):
                if cX.shape[0] < 6:
                    continue
                s, p = cls._single_test(cX, cY, **kwargs)
                p_vals.append(p)
                stats.append(s)
                
            if not p_vals:
                return 0.0, 1.0
        else:
            for _ in range(n_ensembles):
                idx = torch.randperm(n, device=X.device)[:max_samples]
                s, p = cls._single_test(X[idx], Y[idx], **kwargs)
                p_vals.append(p)
                stats.append(s)
            
        return sum(stats) / len(stats), statistics.median(p_vals)

    @classmethod
    def conditional_test(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, max_samples=500, n_ensembles=5, sequential_chunks=False, **kwargs) -> tuple[float, float]:
        """Base conditional test handler to manage dimensional consistency, chunking, and ensembling."""
        X = X.view(-1, 1) if X.ndim == 1 else X
        Y = Y.view(-1, 1) if Y.ndim == 1 else Y
        Z = Z.view(-1, 1) if Z.ndim == 1 else Z
        n = X.shape[0]
        
        if n <= max_samples:
            return cls._single_conditional_test(X, Y, Z, **kwargs)
            
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
                s, p = cls._single_conditional_test(cX, cY, cZ, **kwargs)
                p_vals.append(p)
                stats.append(s)
                
            if not p_vals: 
                return 0.0, 1.0
        else:
            for _ in range(n_ensembles):
                idx = torch.randperm(n, device=X.device)[:max_samples]
                s, p = cls._single_conditional_test(X[idx], Y[idx], Z[idx], **kwargs)
                p_vals.append(p)
                stats.append(s)
            
        return sum(stats) / len(stats), statistics.median(p_vals)