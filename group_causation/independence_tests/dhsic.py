"""Differentiable kernel independence measures (dHSIC).

Unlike :class:`~group_causation.independence_tests.hsic.HSIC_Test`, which
returns p-values, these functions return a torch scalar that keeps the
computation graph, so they can be used as training regularizers for the
iVAE aggregation map.
"""

import torch

from group_causation.independence_tests.hsic import HSIC_Test


def dhsic(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Differentiable multivariate HSIC (dHSIC) statistic between ``X`` and ``Y``.

    Joint Gaussian kernels are used over all columns, so multi-dimensional
    inputs are handled natively. The normalization matches
    ``HSIC_Test._single_test``, up to the constant kernel bandwidths which are
    detached from the graph.
    """
    _, Kc = HSIC_Test.get_gram_matrix(X, HSIC_Test.get_kernel_width(X))
    _, Lc = HSIC_Test.get_gram_matrix(Y, HSIC_Test.get_kernel_width(Y))
    return (Kc * Lc).sum() / X.shape[0]


def pairwise_latent_dhsic(latents: torch.Tensor) -> torch.Tensor:
    """Mean dHSIC over all pairs of latent coordinates.

    Minimizing this term encourages the bottleneck axes to be statistically
    independent (a kernelized disentanglement penalty). Returns zero for a
    single latent coordinate.
    """
    n_dims = latents.shape[1]
    if n_dims < 2:
        return latents.new_zeros(())

    penalties = [
        dhsic(latents[:, [i]], latents[:, [j]])
        for i in range(n_dims)
        for j in range(i + 1, n_dims)
    ]
    return torch.stack(penalties).mean()
