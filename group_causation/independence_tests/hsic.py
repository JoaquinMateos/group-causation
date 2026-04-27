import torch
import math
import statistics
from scipy.stats import gamma

class HSIC_Test:
    """Hilbert-Schmidt Independence Criterion using Gamma approximation (PyTorch Accelerated)."""
    
    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def get_kernel_width(X: torch.Tensor, sample_cut: int = 100) -> float:
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
        n = X.shape[0]
        G = torch.sum(X * X, dim=1)
        H = G.unsqueeze(0) + G.unsqueeze(1) - 2 * (X @ X.T)
        K = torch.exp(-H / (2 * (width**2)))
        
        K_colsums = K.sum(dim=0)
        K_rowsums = K.sum(dim=1)
        K_allsum = K_rowsums.sum()
        Kc = K - (K_colsums.unsqueeze(0) + K_rowsums.unsqueeze(1)) / n + (K_allsum / n**2)
        return K, Kc

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
            # Native PyTorch tensor splitting
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
                # Native PyTorch random indexing without replacement
                idx = torch.randperm(n, device=X.device)[:max_samples]
                s, p = cls._single_test(X[idx], Y[idx])
                p_vals.append(p)
                stats.append(s)
            
        return sum(stats) / len(stats), statistics.median(p_vals)

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

        alpha = mean_val**2 / var_val
        beta = var_val * n / mean_val
        p_val = gamma.sf(test_stat_val, alpha, scale=beta)

        return float(test_stat_val), float(p_val)
    
    @classmethod
    def conditional_test(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, max_samples=500, n_ensembles=5, sequential_chunks=False, epsilon=1e-3) -> tuple[float, float]:
        X = X.view(-1, 1) if X.ndim == 1 else X
        Y = Y.view(-1, 1) if Y.ndim == 1 else Y
        Z = Z.view(-1, 1) if Z.ndim == 1 else Z
        n = X.shape[0]
        
        if n <= max_samples:
            return cls._single_conditional_test(X, Y, Z, epsilon)
            
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
                s, p = cls._single_conditional_test(cX, cY, cZ, epsilon)
                p_vals.append(p)
                stats.append(s)
                
            if not p_vals: 
                return 0.0, 1.0
        else:
            for _ in range(n_ensembles):
                idx = torch.randperm(n, device=X.device)[:max_samples]
                s, p = cls._single_conditional_test(X[idx], Y[idx], Z[idx], epsilon)
                p_vals.append(p)
                stats.append(s)
            
        return sum(stats) / len(stats), statistics.median(p_vals)

    @classmethod
    def _single_conditional_test(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, epsilon: float = 1e-3) -> tuple[float, float]:
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

        I = torch.eye(n, dtype=torch.float64, device=X.device)
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
        var_approx = (2 / n**2) * torch.sum(eig_x**2).item() * torch.sum(eig_y**2).item()

        if var_approx <= 0 or mean_approx <= 0:
            return float(test_stat), 1.0

        alpha = (mean_approx**2) / var_approx
        beta = var_approx / mean_approx
        p_val = gamma.sf(test_stat, alpha, scale=beta)

        return float(test_stat), float(p_val)