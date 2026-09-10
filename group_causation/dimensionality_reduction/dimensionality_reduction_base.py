from abc import ABC, abstractmethod
from typing import Any
import torch

class DimensionalityReduction(ABC):
    """Abstract base class unifying dimensionality reduction techniques."""

    @abstractmethod
    def fit(self, X: Any, U: Any | None = None, **kwargs) -> 'DimensionalityReduction':
        pass

    @abstractmethod
    def transform(self, X: Any, U: Any | None = None, **kwargs) -> Any:
        pass

    def fit_transform(self, X: Any, U: Any | None = None, **kwargs) -> Any:
        self.fit(X, U, **kwargs)
        return self.transform(X, U, **kwargs)