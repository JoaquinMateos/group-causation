import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import gamma
from typing import Union
from abc import abstractmethod

# Assuming GroupCausalDiscovery is defined elsewhere in your project
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery


# ---------------------------------------------------------------------------
# 1. HSIC Independence Test (Unchanged - Used for Phase I)
# ---------------------------------------------------------------------------
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
    def test(cls, X: np.ndarray, Y: np.ndarray, max_samples=500, n_ensembles=5) -> tuple[float, float]:
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        n = X.shape[0]
        
        if n <= max_samples:
            return cls._single_test(X, Y)
            
        p_vals = []
        stats = []
        
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

# ---------------------------------------------------------------------------
# 2. MLP Regressors (Standard & Spatio-Temporal MURGS)
# ---------------------------------------------------------------------------
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class GroupRegressor:
    """Standard Regressor used for Phase I residual computation."""
    def __init__(self, epochs=200, batch_size=200, lr=0.01, hidden_dim=100):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dim = hidden_dim

    def fit(self, X: np.ndarray, Y: np.ndarray):
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiOutputMLP(input_dim, output_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.FloatTensor(X).to(self.device))
        return preds.cpu().numpy()

class SpatioTemporalMURGSRegressor(GroupRegressor):
    """
    Regressor with L2,1 Group Lasso Penalty on the input layer weights 
    to implement the Temporal-MURGS pruning for Phase II.
    """
    def __init__(self, epochs=200, batch_size=200, lr=0.01, hidden_dim=100, lambda_reg=0.01):
        super().__init__(epochs, batch_size, lr, hidden_dim)
        self.lambda_reg = lambda_reg

    def fit(self, X: np.ndarray, Y: np.ndarray, group_dims: list[int]):
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiOutputMLP(input_dim, output_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                preds = self.model(batch_x)
                mse_loss = criterion(preds, batch_y)
                
                # Apply Spatio-Temporal Group Lasso (MURGS Penalty)
                reg_loss = 0.0
                start_idx = 0
                W_in = self.model.net[0].weight  # Shape: (hidden_dim, input_dim)
                
                for g_dim in group_dims:
                    end_idx = start_idx + g_dim
                    W_group = W_in[:, start_idx:end_idx]
                    # MURGS functional: sqrt(d_g) * ||W_g||_F
                    reg_loss += math.sqrt(g_dim) * torch.norm(W_group, p='fro')
                    start_idx = end_idx
                
                loss = mse_loss + self.lambda_reg * reg_loss
                loss.backward()
                optimizer.step()

    def get_group_norms(self, group_dims: list[int]) -> list[float]:
        """Returns the Frobenius norm of the input weights associated with each feature group."""
        self.model.eval()
        norms = []
        start_idx = 0
        with torch.no_grad():
            W_in = self.model.net[0].weight.cpu()
            for g_dim in group_dims:
                end_idx = start_idx + g_dim
                W_group = W_in[:, start_idx:end_idx]
                norms.append(float(torch.norm(W_group, p='fro')))
                start_idx = end_idx
        return norms

# ---------------------------------------------------------------------------
# 3. Time-Series GroupRESIT-MURGS Algorithm
# ---------------------------------------------------------------------------
class GroupRESITTimeSeriesCausalDiscovery(GroupCausalDiscovery):
    '''
    Time-Series adaptation of the GroupRESIT Algorithm.
    Phase I: HSIC-based Sink Node identification for contemporaneous order.
    Phase II: Spatio-Temporal MURGS pruning via Group-Lasso Neural Networks.
    '''
    def __init__(self, data: np.ndarray, groups: Union[list[set[int]], None] = None,
                 standarize: bool=True, **kwargs):
        super().__init__(data, groups, standarize, **kwargs)
        
        # Hyperparameters
        self.epochs = self.extra_args.get("epochs", 200)
        self.hidden_dim = self.extra_args.get("hidden_dim", 100)
        self.max_lag = self.extra_args.get("max_lag", 1)
        self.min_lag = self.extra_args.get("min_lag", 1) 
        
        # MURGS specific hyperparameters
        self.lambda_reg = self.extra_args.get("lambda_reg", 0.05) # Regularization strength
        self.pruning_threshold = self.extra_args.get("pruning_threshold", 1e-3) # Threshold to drop edge
        
        self.T = self._data.shape[0]
        self.G = len(self._groups)
        
        if self.T <= self.max_lag:
            raise ValueError("Time series length T must be strictly greater than max_lag.")
        if self.min_lag > self.max_lag:
            raise ValueError("min_lag cannot be strictly greater than max_lag.")
        
        # Internal state
        self._causal_order = [] 
        self._pa = {}           

    def _get_data_and_dims_for_vars(self, vars_list: list[tuple[int, int]]) -> tuple[np.ndarray, list[int]]:
        """
        Constructs a flat 2D array of specific groups at specific lags,
        and returns the feature dimensions of each group block for the MURGS penalty.
        """
        if not vars_list:
            return np.empty((self.T - self.max_lag, 0)), []
            
        blocks = []
        dims = []
        for g, l in vars_list:
            cols = list(self._groups[g]) # Ensure it's list-like for indexing
            start_idx = self.max_lag - l
            end_idx = self.T - l
            blocks.append(self._data[start_idx:end_idx, cols])
            dims.append(len(cols))
            
        return np.concatenate(blocks, axis=1), dims

    def _phase_1_causal_order(self):
        """Phase I: Infer the causal order among contemporary variables (lag 0)."""
        if self.min_lag > 0:
            self._causal_order = list(range(self.G))
            return

        S = list(range(self.G))
        pi_contemp = []
        
        start_lag = max(1, self.min_lag)
        past_vars = [(g, l) for g in range(self.G) for l in range(start_lag, self.max_lag + 1)]

        while S:
            if len(S) == 1:
                pi_contemp.insert(0, S[0])
                break

            best_group = None
            least_dependent_stat = float('inf')

            for g in S:
                rem_contemp = [(rem_g, 0) for rem_g in S if rem_g != g]
                regressors = rem_contemp + past_vars
                
                Y, _ = self._get_data_and_dims_for_vars([(g, 0)])
                
                if not regressors:
                    least_dependent_stat = 0.0
                    best_group = g
                    break
                    
                X, _ = self._get_data_and_dims_for_vars(regressors)

                # Standard Regression for conditional independence testing
                regressor = GroupRegressor(epochs=self.epochs, hidden_dim=self.hidden_dim)
                regressor.fit(X, Y)
                Y_pred = regressor.predict(X)
                
                residuals = Y - Y_pred
                residuals = (residuals - residuals.mean(axis=0)) / (residuals.std(axis=0) + 1e-8)
                X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
                
                test_stat, p_val = HSIC_Test.test(residuals, X_std)
                
                if test_stat < least_dependent_stat:
                    least_dependent_stat = test_stat
                    best_group = g

            if best_group is None:
                raise RuntimeError("Failed to determine causal order in Phase I.")
            
            S.remove(best_group)
            pi_contemp.insert(0, best_group)

        self._causal_order = pi_contemp

    def _phase_2_pruning(self):
        """Phase II: Spatio-Temporal MURGS Model Selection."""
        pa = {}
        
        start_lag = max(1, self.min_lag)
        past_vars = [(g, l) for g in range(self.G) for l in range(start_lag, self.max_lag + 1)]

        for i, k in enumerate(self._causal_order):
            contemp_preds = [(p, 0) for p in self._causal_order[:i]] if self.min_lag == 0 else []
            potential_parents = contemp_preds + past_vars
            
            if not potential_parents:
                pa[k] = []
                continue

            # Extract data and group dimensions for all potential parents simultaneously
            X_pot_parents, group_dims = self._get_data_and_dims_for_vars(potential_parents)
            Y, _ = self._get_data_and_dims_for_vars([(k, 0)])
            
            # Standardize
            X_pot_parents = (X_pot_parents - X_pot_parents.mean(axis=0)) / (X_pot_parents.std(axis=0) + 1e-8)
            Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)

            # Fit MURGS model with group-lasso penalty across space and time
            murgs_model = SpatioTemporalMURGSRegressor(
                epochs=self.epochs, 
                hidden_dim=self.hidden_dim, 
                lambda_reg=self.lambda_reg
            )
            murgs_model.fit(X_pot_parents, Y, group_dims)
            
            # Extract norms to filter connections
            norms = murgs_model.get_group_norms(group_dims)
            
            # Keep parents whose weight block survived the penalty shrinkage
            surviving_parents = []
            for p_idx, norm_val in enumerate(norms):
                if norm_val > self.pruning_threshold:
                    surviving_parents.append(potential_parents[p_idx])
                    
            pa[k] = surviving_parents

        self._pa = pa

    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        """
        Execute Phase I & Phase II, returning the DAG as a dictionary 
        mapping node -> list of parent nodes with their respective time lags.
        """
        self._phase_1_causal_order()
        self._phase_2_pruning()
        
        final_parents = {i: [] for i in range(self.G)}
        
        for node, parents in self._pa.items():
            formatted_parents = []
            for p, l in parents:
                formatted_lag = -l if l > 0 else 0
                formatted_parents.append((p, formatted_lag))
            final_parents[node] = formatted_parents
            
        return final_parents