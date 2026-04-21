from typing import Union

import logging
import numpy as np
from scipy.stats import ks_2samp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery

# ---------------------------------------------------------------------------
# 1. Knockoff Generator (Gaussian Second-Order)
# ---------------------------------------------------------------------------
class GaussianKnockoffGenerator:
    """
    Generates Second-Order Gaussian Knockoffs.
    Knockoffs are in-distribution variables that preserve the covariance 
    structure but are uncorrelated with the original variables.
    """
    @staticmethod
    def generate(X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        mu = np.mean(X, axis=0)
        X_centered = X - mu
        
        # Empirical covariance
        Sigma = np.cov(X_centered, rowvar=False)
        if p == 1:
            Sigma = np.array([[np.var(X_centered)]])
        
        # Add slight jitter for numerical stability (positive definiteness)
        Sigma += np.eye(p) * 1e-6
        
        eigenvalues = np.linalg.eigvalsh(Sigma)
        lambda_min = min(eigenvalues)
        
        # S matrix: diagonal matrix with s_i. Ensure 2*Sigma - S is PSD.
        s = min(1.0, 2 * lambda_min) * 0.99
        S = np.eye(p) * s
        
        # Calculate knockoff distribution parameters
        Sigma_inv = np.linalg.inv(Sigma)
        mu_tilde = X_centered - X_centered @ Sigma_inv @ S
        V = 2 * S - S @ Sigma_inv @ S
        
        # Sample knockoffs
        noise = np.random.multivariate_normal(np.zeros(p), V, size=n)
        X_knockoff = mu_tilde + noise + mu
        
        return X_knockoff

# ---------------------------------------------------------------------------
# 2. DeepAR Probabilistic Forecaster (PyTorch Implementation)
# ---------------------------------------------------------------------------
class DeepAR(nn.Module):
    """
    Deep Autoregressive Recurrent Network for Probabilistic Forecasting.
    Models the temporal dynamics of the system and outputs parameters 
    for a Gaussian distribution (mu, sigma) at the next time step.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Projection layers to map LSTM hidden state to distribution parameters
        self.mu_layer = nn.Linear(hidden_dim, input_dim)
        self.sigma_layer = nn.Linear(hidden_dim, input_dim)
        
        # Softplus ensures standard deviation is strictly positive
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, features)
        Returns mu and sigma for the time step immediately following the sequence.
        """
        lstm_out, _ = self.lstm(x)
        
        # We only care about predicting the next step after the historical window
        last_hidden_state = lstm_out[:, -1, :]
        
        mu = self.mu_layer(last_hidden_state)
        # Add epsilon to prevent sigma from becoming exactly zero
        sigma = self.softplus(self.sigma_layer(last_hidden_state)) + 1e-6 
        
        return mu, sigma

def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the Negative Log-Likelihood of a Gaussian Distribution."""
    distribution = torch.distributions.Normal(mu, sigma)
    # We want to minimize the negative log probability
    return -distribution.log_prob(target).mean()

# ---------------------------------------------------------------------------
# 3. gCDMI Algorithm
# ---------------------------------------------------------------------------
class gCDMICausalDiscovery(GroupCausalDiscovery):
    '''
    Group Interventions on Deep Networks for Causal Discovery (gCDMI).
    Uses a DeepAR formulation for structure learning, group-wise knockoff interventions,
    and infers causality via Model Invariance Testing (KS Test).
    '''
    def __init__(self, data: np.ndarray, groups: Union[list[set[int]], None] = None,
                 standarize: bool=True, **kwargs):
        super().__init__(data, groups, standarize, **kwargs)
        
        # Hyperparameters
        self.alpha = self.extra_args.get("alpha", 0.05) # Significance level for KS-Test
        self.epochs = self.extra_args.get("epochs", 150)
        self.hidden_dim = self.extra_args.get("hidden_dim", 64)
        self.num_layers = self.extra_args.get("num_layers", 2)
        self.batch_size = self.extra_args.get("batch_size", 128)
        self.lr = self.extra_args.get("learning_rate", 0.005)
        self.max_lag = self.extra_args.get("max_lag", 3) # DeepAR benefits from longer context
        self.lambda_l1 = self.extra_args.get("lambda_l1", 1e-4) # L1 regularization for sparsity in the learned structure
        
        self.T, self.N = self._data.shape
        self.G = len(self._groups)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        if self.T <= self.max_lag:
            raise ValueError("Time series length T must be strictly greater than max_lag.")

    def _create_windows(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Creates sliding autoregressive windows for DeepAR forecasting."""
        X, Y = [], []
        for i in range(self.T - self.max_lag):
            X.append(data[i : i + self.max_lag, :])
            Y.append(data[i + self.max_lag, :])
        return np.array(X), np.array(Y)

    def _train_structure(self):
        """Step 1: Structure Learning. Train DeepAR to forecast the multivariate system.
        Uses early stopping based on validation loss to prevent overfitting and ensure better generalization.
        """        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepAR(
            input_dim=self.N, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        
        X_seq, Y_seq = self._create_windows(self._data)
        
        # --- SPLIT 80/20 ---
        split_idx = int(len(X_seq) * 0.8)
        X_train, Y_train = X_seq[:split_idx], Y_seq[:split_idx]
        X_val, Y_val = X_seq[split_idx:], Y_seq[split_idx:]
        
        dataset_train = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        Y_val_t = torch.FloatTensor(Y_val).to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15

        for epoch in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in loader_train:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                mu, sigma = self.model(batch_x)
                
                loss = gaussian_nll_loss(mu, sigma, batch_y)
                l1_reg = torch.norm(self.model.lstm.weight_ih_l0, 1)
                loss += self.lambda_l1 * l1_reg
                
                loss.backward()
                optimizer.step()
                
            # --- VALIDATION ---
            self.model.eval()
            with torch.no_grad():
                mu_val, sigma_val = self.model(X_val_t)
                val_loss = gaussian_nll_loss(mu_val, sigma_val, Y_val_t).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.info(f"Early stopping en epoch {epoch} (Val Loss no mejora).")
                break
                
        # Restaurar el modelo que mejor predice la realidad
        if 'best_weights' in locals():
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})

    def _compute_residuals(self, true_y: np.ndarray, pred_mu: np.ndarray, group_idx: int) -> np.ndarray:
        """
        Calculates the relative absolute errors for a specific group based on the 
        DeepAR point forecast (mu).
        Formula from paper: e = |Z - Z_hat| / |Z|
        """
        cols = list(self._groups[group_idx])
        Z = true_y[:, cols]
        Z_hat = pred_mu[:, cols]
        
        # Add epsilon to denominator to prevent division by zero in zero-scaled data
        residuals = np.abs(Z - Z_hat) / (np.abs(Z) + 1e-8)
        return residuals.flatten() # Flatten to 1D distribution array for the KS-test

    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        """
        Execute Steps 1 (Learning), 2 (Interventions), and 3 (Invariance Testing).
        """
        # Step 1: Learn the structural temporal dependencies via DeepAR
        self._train_structure()
        
        # Generate knockoffs for the entire dataset
        self._knockoffs = GaussianKnockoffGenerator.generate(self._data)
        
        # Get baseline (observational) sequences and their true targets
        X_obs, Y_true = self._create_windows(self._data)
        
        self.model.eval()
        with torch.no_grad():
            mu_obs, _ = self.model(torch.FloatTensor(X_obs).to(self.device))
            Y_pred_obs = mu_obs.cpu().numpy()

        causal_graph = {i: [] for i in range(self.G)}

        # Step 2 & 3: Group Interventions & Model Invariance Test
        for i in range(self.G): # Candidate cause group
            
            # Create interventional data: Replace group `i` entirely with its knockoff
            interventional_data = self._data.copy()
            cols_i = self._groups[i]
            interventional_data[:, cols_i] = self._knockoffs[:, cols_i]
            
            # Reconstruct the temporal windows using the interventional sequence
            X_interv, _ = self._create_windows(interventional_data)
            
            with torch.no_grad():
                mu_interv, _ = self.model(torch.FloatTensor(X_interv).to(self.device))
                Y_pred_interv = mu_interv.cpu().numpy()
                
            for j in range(self.G): # Target group    
                # Extract residuals for target group j
                R_j = self._compute_residuals(Y_true, Y_pred_obs, j)
                R_j_tilde = self._compute_residuals(Y_true, Y_pred_interv, j)
                
                # --- NUEVO: Submuestreo para evitar la hipersensibilidad del KS-Test ---
                max_samples = min(400, len(R_j))
                idx = np.random.choice(len(R_j), max_samples, replace=False)
                
                # Kolmogorov-Smirnov Test en el subconjunto
                stat, p_val = ks_2samp(R_j[idx], R_j_tilde[idx])
                
                # If p-value < alpha, we reject the null hypothesis (invariance broken).
                # Therefore, group i has a causal effect on group j.
                if p_val < self.alpha:
                    # Time-series causality is interpreted as the past of i causes the present of j.
                    # We format this as a lag of -1 to match the evaluation framework expectations.
                    causal_graph[j].append((i, -1))

        return causal_graph