"""Hilbert-Schmidt Independence Criterion using Gamma approximation (PyTorch Accelerated)."""

import torch
from scipy.stats import gamma

from group_causation.independence_tests.conditional_independence_base import (
    ConditionalIndependence_base,
    _get_device,
)


class HSIC_Test(ConditionalIndependence_base):
    """HSIC independence test with Gamma approximation for p-values."""

    # ------------------------------------------------------------------
    # Kernel utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_kernel_width(X: torch.Tensor, sample_cut: int = 100) -> float:
        """Median heuristic for the Gaussian kernel width."""
        n_samples = X.shape[0]
        if n_samples > sample_cut:
            X_med = X[:sample_cut, :]
            n_samples = sample_cut
        else:
            X_med = X

        G = torch.sum(X_med * X_med, dim=1).reshape(n_samples, 1)
        dists = G + G.T - 2 * (X_med @ X_med.T)
        dists = dists - torch.tril(dists)
        dists = dists.reshape(-1)

        pos_dists = dists[dists > 0]
        if len(pos_dists) > 0:
            med = torch.median(pos_dists).item()
            return (0.5 * med) ** 0.5 if med > 0 else 1.0
        return 1.0

    @staticmethod
    def get_gram_matrix(X: torch.Tensor, width: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the kernel matrix K and its centred version Kc."""
        n = X.shape[0]
        G = torch.sum(X * X, dim=1)
        H = G.unsqueeze(0) + G.unsqueeze(1) - 2 * (X @ X.T)
        K = torch.exp(-H / (2 * (width ** 2)))

        K_colsums = K.sum(dim=0)
        K_rowsums = K.sum(dim=1)
        K_allsum = K_rowsums.sum()
        Kc = K - (K_colsums.unsqueeze(0) + K_rowsums.unsqueeze(1)) / n + (K_allsum / n ** 2)
        return K, Kc

    # ------------------------------------------------------------------
    # Required hooks from ConditionalIndependence_base
    # ------------------------------------------------------------------

    @classmethod
    def _single_test(cls, X: torch.Tensor, Y: torch.Tensor) -> tuple[float, float]:
        n = X.shape[0]
        if n < 6:
            return 0.0, 1.0

        width_x = cls.get_kernel_width(X)
        width_y = cls.get_kernel_width(Y)

        K, Kc = cls.get_gram_matrix(X, width_x)
        L, Lc = cls.get_gram_matrix(Y, width_y)

        test_stat = (1 / n) * torch.sum(Kc.T * Lc)

        var = (1 / 6 * Kc * Lc) ** 2
        var = (1 / (n * (n - 1))) * (torch.sum(var) - torch.trace(var))
        var = 72 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3)) * var

        K.fill_diagonal_(0)
        L.fill_diagonal_(0)
        mu_X = 1 / (n * (n - 1)) * K.sum()
        mu_Y = 1 / (n * (n - 1)) * L.sum()

        mean = 1 / n * (1 + mu_X * mu_Y - mu_X - mu_Y)

        test_stat_val = test_stat.item()
        mean_val = mean.item()
        var_val = var.item()

        if var_val <= 0 or mean_val <= 0:
            return float(test_stat_val), 1.0

        alpha = mean_val ** 2 / var_val
        beta = var_val * n / mean_val
        p_val = gamma.sf(test_stat_val, alpha, scale=beta)

        return float(test_stat_val), float(p_val)

    @classmethod
    def _single_conditional_test(cls, X: torch.Tensor, Y: torch.Tensor,
                                 Z: torch.Tensor, epsilon: float = 1e-3) -> tuple[float, float]:
        n = X.shape[0]
        if n < 6:
            return 0.0, 1.0

        wx = cls.get_kernel_width(X)
        wy = cls.get_kernel_width(Y)
        wz = cls.get_kernel_width(Z)

        _, Kc_X = cls.get_gram_matrix(X, wx)
        _, Kc_Y = cls.get_gram_matrix(Y, wy)
        _, Kc_Z = cls.get_gram_matrix(Z, wz)

        Kc_X = (Kc_X + Kc_X.T) / 2
        Kc_Y = (Kc_Y + Kc_Y.T) / 2
        Kc_Z = (Kc_Z + Kc_Z.T) / 2

        I = torch.eye(n, dtype=X.dtype, device=X.device)
        scaled_epsilon = epsilon * n
        P_z = scaled_epsilon * torch.linalg.inv(Kc_Z + scaled_epsilon * I)
        P_z = (P_z + P_z.T) / 2

        K_xz = P_z @ Kc_X @ P_z
        K_yz = P_z @ Kc_Y @ P_z

        K_xz = (K_xz + K_xz.T) / 2
        K_yz = (K_yz + K_yz.T) / 2

        test_stat = torch.sum(K_xz * K_yz).item()

        eig_x = torch.linalg.eigh(K_xz)[0]
        eig_y = torch.linalg.eigh(K_yz)[0]

        max_x, max_y = torch.max(eig_x), torch.max(eig_y)

        if max_x <= 0 or max_y <= 0:
            return float(test_stat), 1.0

        eig_x = eig_x[eig_x > max_x * 1e-5]
        eig_y = eig_y[eig_y > max_y * 1e-5]

        if len(eig_x) == 0 or len(eig_y) == 0:
            return float(test_stat), 1.0

        mean_approx = (1 / n) * torch.sum(eig_x).item() * torch.sum(eig_y).item()
        var_approx = (2 / n ** 2) * torch.sum(eig_x ** 2).item() * torch.sum(eig_y ** 2).item()

        if var_approx <= 0 or mean_approx <= 0:
            return float(test_stat), 1.0

        alpha = (mean_approx ** 2) / var_approx
        beta = var_approx / mean_approx
        p_val = gamma.sf(test_stat, alpha, scale=beta)

        return float(test_stat), float(p_val)
