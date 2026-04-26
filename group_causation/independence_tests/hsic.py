import numpy as np
from scipy.stats import gamma

class HSIC_Test:
    """Hilbert-Schmidt Independence Criterion using Gamma approximation."""
    
    @staticmethod
    def get_kernel_width(X: np.ndarray, sample_cut: int = 100) -> float:
        n_samples = X.shape[0]
        if n_samples > sample_cut:
            X_med = X[:sample_cut, :]
            n_samples = sample_cut
        else:
            X_med = X

        G = np.sum(X_med * X_med, 1).reshape(n_samples, 1)
        dists = G + G.T - 2 * np.dot(X_med, X_med.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n_samples**2, 1)
        med = np.median(dists[dists > 0])
        return np.sqrt(0.5 * med) if med > 0 else 1.0

    @staticmethod
    def get_gram_matrix(X: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        G = np.sum(X * X, axis=1)
        H = G[None, :] + G[:, None] - 2 * np.dot(X, X.T)
        K = np.exp(-H / (2 * (width**2)))
        
        K_colsums = K.sum(axis=0)
        K_rowsums = K.sum(axis=1)
        K_allsum = K_rowsums.sum()
        Kc = K - (K_colsums[None, :] + K_rowsums[:, None]) / n + (K_allsum / n**2)
        return K, Kc

    @classmethod
    def test(cls, X: np.ndarray, Y: np.ndarray, max_samples=500, n_ensembles=5, sequential_chunks=False) -> tuple[float, float]:
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        n = X.shape[0]
        
        if n <= max_samples:
            return cls._single_test(X, Y)
            
        p_vals = []
        stats = []
        
        if sequential_chunks:
            # Divide into sequential blocks to preserve local temporal stationarity
            num_chunks = max(1, int(np.ceil(n / max_samples)))
            chunk_indices = np.array_split(np.arange(n), num_chunks)
            
            for idx in chunk_indices:
                if len(idx) < 6: # Skip tiny chunks that break HSIC math
                    continue
                s, p = cls._single_test(X[idx], Y[idx])
                p_vals.append(p)
                stats.append(s)
                
            if not p_vals: # Fallback if no valid chunks exist
                return 0.0, 1.0
        else:
            # Random uniform subsampling
            for _ in range(n_ensembles):
                idx = np.random.choice(n, max_samples, replace=False)
                s, p = cls._single_test(X[idx], Y[idx])
                p_vals.append(p)
                stats.append(s)
            
        return float(np.mean(stats)), float(np.median(p_vals))

    @classmethod
    def _single_test(cls, X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        n = X.shape[0]
        
        if n < 6:
            return 0.0, 1.0 

        width_x = cls.get_kernel_width(X)
        width_y = cls.get_kernel_width(Y)

        K, Kc = cls.get_gram_matrix(X, width_x)
        L, Lc = cls.get_gram_matrix(Y, width_y)

        test_stat = (1 / n) * np.sum(Kc.T * Lc)

        var = (1 / 6 * Kc * Lc) ** 2
        var = (1 / (n * (n - 1))) * (np.sum(var) - np.trace(var))
        var = 72 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3)) * var

        K[np.diag_indices(n)] = 0
        L[np.diag_indices(n)] = 0
        mu_X = 1 / (n * (n - 1)) * K.sum()
        mu_Y = 1 / (n * (n - 1)) * L.sum()
        
        mean = 1 / n * (1 + mu_X * mu_Y - mu_X - mu_Y)
        
        if var <= 0 or mean <= 0:
            return float(test_stat), 1.0

        alpha = mean**2 / var
        beta = var * n / mean
        p_val = gamma.sf(test_stat, alpha, scale=beta)

        return float(test_stat), float(p_val)