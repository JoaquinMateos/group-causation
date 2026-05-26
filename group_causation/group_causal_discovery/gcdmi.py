from typing import Union, Optional, Any
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
class TimeSeriesKnockoffGenerator:
    """
    Generates Time-Series Knockoffs using Fourier Phase Randomization.
    Preserves autocorrelation (power spectrum) while destroying causal cross-correlations.
    """
    @staticmethod
    def generate(X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        X_knockoff = np.zeros_like(X)
        
        for i in range(p):
            # 1. Transform to frequency domain
            fft_coeffs = np.fft.rfft(X[:, i])
            
            # 2. Extract amplitudes and phases
            amplitudes = np.abs(fft_coeffs)
            phases = np.angle(fft_coeffs)
            
            # 3. Randomize phases (keep the first phase 0 for zero-frequency/DC component)
            random_phases = np.random.uniform(0, 2 * np.pi, len(phases))
            random_phases[0] = phases[0] 
            
            # 4. Reconstruct complex coefficients
            new_coeffs = amplitudes * np.exp(1j * random_phases)
            
            # 5. Inverse FFT back to time domain
            X_knockoff[:, i] = np.fft.irfft(new_coeffs, n=n)
            
        return X_knockoff

# ---------------------------------------------------------------------------
# 2. DeepAR Probabilistic Forecaster (PyTorch Implementation)
# ---------------------------------------------------------------------------
class DeepAR(nn.Module):
    """
    Deep Autoregressive Recurrent Network for Probabilistic Forecasting.
    Models the temporal dynamics of the system and outputs parameters 
    for a Gaussian distribution (mu, sigma) at the next time step.
    Accepts background variables (u) concatenated to inputs if provided.
    """
    def __init__(self, input_dim: int, u_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim + u_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.mu_layer = nn.Linear(hidden_dim, input_dim)
        self.sigma_layer = nn.Linear(hidden_dim, input_dim)
        self.softplus = nn.Softplus()

    def forward(self, x, u=None):
        """
        x shape: (batch_size, sequence_length, features)
        u shape: (batch_size, sequence_length, u_dim) - optional
        """
        if u is not None:
            x = torch.cat([x, u], dim=-1)
            
        lstm_out, _ = self.lstm(x)
        
        mu = self.mu_layer(lstm_out)
        sigma = self.softplus(self.sigma_layer(lstm_out)) + 1e-6 
        
        return mu, sigma

def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distribution = torch.distributions.Normal(mu, sigma)
    return -distribution.log_prob(target).mean()

# ---------------------------------------------------------------------------
# 3. gCDMI Algorithm
# ---------------------------------------------------------------------------
class gCDMICausalDiscovery(GroupCausalDiscovery):
    '''
    Group Interventions on Deep Networks for Causal Discovery (gCDMI).
    Uses a DeepAR formulation for structure learning, group-wise knockoff interventions,
    and infers causality via Model Invariance Testing (KS Test).
    Can optionally incorporate background variables to handle non-stationarity.
    '''
    def __init__(self, data: np.ndarray, groups: Union[list[set[int]], None] = None,
                 standarize: bool=True, non_stationarity_info: Union[dict, None] = None, 
                 use_nonstationarity_info: bool = False,
                 verbose: int = 0, **kwargs):
                 
        super().__init__(data, groups, standarize, **kwargs)
        
        self.alpha = self.extra_args.get("alpha", 0.05)
        self.epochs = self.extra_args.get("epochs", 150)
        self.hidden_dim = self.extra_args.get("hidden_dim", 64)
        self.num_layers = self.extra_args.get("num_layers", 2)
        self.batch_size = self.extra_args.get("batch_size", 128)
        self.lr = self.extra_args.get("learning_rate", 0.005)
        self.max_lag = self.extra_args.get("max_lag", 3)
        self.lambda_l1 = self.extra_args.get("lambda_l1", 1e-4)
        self._verbose = verbose
        
        self.T, self.N = self._data.shape
        self.G = len(self._groups)
        
        if self.T <= self.max_lag:
            raise ValueError("Time series length T must be strictly greater than max_lag.")
            
        # ---------------------------------------------------------
        # Conditional Background 'u' Construction
        # ---------------------------------------------------------
        self.use_nonstationarity_info = use_nonstationarity_info
        self.u = None
        
        if self.use_nonstationarity_info:
            non_stationarity_info = non_stationarity_info if non_stationarity_info is not None else {}
            
            if non_stationarity_info.get('type') != 'regime_shifts':
                raise ValueError("non_stationarity_info must have type 'regime_shifts' when use_nonstationarity_info=True")
            
            affected_vars = non_stationarity_info.get('affected_vars', [])
            if not affected_vars:
                if self._verbose > 0:
                    logging.info("Notice: No variables affected by non-stationarity. Falling back to default model.")
                self.use_nonstationarity_info = False
            else:
                first_var = affected_vars[0]
                shifts = non_stationarity_info['shift_details'][first_var]
                
                total_T = shifts[-1]['end']
                u_full = np.zeros(total_T, dtype=int)
                
                for shift in shifts:
                    u_full[shift['start']:shift['end']] = shift['regime'] - 1
                
                u_aligned = u_full[-self.T:]
                
                num_regimes = non_stationarity_info.get('num_shifts', len(shifts)) + 1
                u_one_hot = np.zeros((self.T, num_regimes))
                u_one_hot[np.arange(self.T), u_aligned] = 1
                
                self.u = u_one_hot

        self.u_dim = self.u.shape[1] if self.u is not None else 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def _create_windows(self, data: np.ndarray, u_data: Optional[np.ndarray] = None) -> tuple:
        """Creates sliding autoregressive windows for DeepAR forecasting."""
        X, Y = [], []
        U = [] if u_data is not None else None
        
        for i in range(len(data) - self.max_lag):
            X.append(data[i : i + self.max_lag, :])
            Y.append(data[i + 1 : i + self.max_lag + 1, :])
            if U is not None and u_data is not None:
                U.append(u_data[i : i + self.max_lag, :])
                
        if U is not None:
            return np.array(X), np.array(U), np.array(Y)
        return np.array(X), np.array(Y)

    def _train_structure(self):
        """Step 1: Structure Learning. Train DeepAR to forecast the multivariate system."""        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DeepAR(
            input_dim=self.N, 
            u_dim=self.u_dim,
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.use_nonstationarity_info:
            X_seq, U_seq, Y_seq = self._create_windows(self._data, self.u)
        else:
            X_seq, Y_seq = self._create_windows(self._data)
            
        # --- SPLIT 80/20 ---
        split_idx = int(len(X_seq) * 0.8)
        X_train, Y_train = X_seq[:split_idx], Y_seq[:split_idx]
        X_val, Y_val = X_seq[split_idx:], Y_seq[split_idx:]
        
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        Y_val_t = torch.FloatTensor(Y_val).to(self.device)
        
        if self.use_nonstationarity_info:
            U_train, U_val = U_seq[:split_idx], U_seq[split_idx:]
            dataset_train = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(U_train), torch.FloatTensor(Y_train))
            U_val_t = torch.FloatTensor(U_val).to(self.device)
        else:
            dataset_train = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
            U_val_t = None
            
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15

        for epoch in range(self.epochs):
            self.model.train()
            
            for batch in loader_train:
                if self.use_nonstationarity_info:
                    batch_x, batch_u, batch_y = batch
                    batch_u = batch_u.to(self.device)
                else:
                    batch_x, batch_y = batch
                    batch_u = None
                    
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                mu, sigma = self.model(batch_x, batch_u)
                
                loss = gaussian_nll_loss(mu, sigma, batch_y)
                l1_reg = torch.norm(self.model.lstm.weight_ih_l0, 1)
                loss += self.lambda_l1 * l1_reg
                
                loss.backward()
                optimizer.step()
                
            # --- VALIDATION ---
            self.model.eval()
            with torch.no_grad():
                mu_val, sigma_val = self.model(X_val_t, U_val_t)
                val_loss = gaussian_nll_loss(mu_val, sigma_val, Y_val_t).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.debug(f"Early stopping en epoch {epoch} (Val Loss no mejora).")
                break
                
        if 'best_weights' in locals():
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})

    def _compute_residuals(self, true_y: np.ndarray, pred_mu: np.ndarray, group_idx: int) -> np.ndarray:
        cols = list(self._groups[group_idx])
        Z = true_y[:, :, cols]       # Shape: (batch, time, features)
        Z_hat = pred_mu[:, :, cols]  # Shape: (batch, time, features)
        
        # Calculate step-by-step relative error
        step_errors = np.abs(Z - Z_hat) / (np.abs(Z) + 1e-8)
        
        # Average over the time dimension (axis 1)
        window_residuals = np.mean(step_errors, axis=1)
        
        return window_residuals.flatten()

    def score_validation_nll(self, validation_data: np.ndarray, validation_u: Optional[np.ndarray] = None) -> float:
        """Scores the trained DeepAR model on a held-out time-series split using Gaussian NLL."""
        if not hasattr(self, 'model'):
            raise RuntimeError('The gCDMI model must be trained before calling score_validation_nll().')

        if self.use_nonstationarity_info and validation_u is None:
            validation_u = self.u[-validation_data.shape[0]:] if self.u is not None else None

        if self.use_nonstationarity_info:
            X_seq, U_seq, Y_seq = self._create_windows(validation_data, validation_u)
            U_seq_t = torch.FloatTensor(U_seq).to(self.device)
        else:
            X_seq, Y_seq = self._create_windows(validation_data)
            U_seq_t = None

        if len(X_seq) == 0:
            return float('inf')

        with torch.no_grad():
            mu_val, sigma_val = self.model(torch.FloatTensor(X_seq).to(self.device), U_seq_t)
            Y_val_t = torch.FloatTensor(Y_seq).to(self.device)
            val_loss = gaussian_nll_loss(mu_val, sigma_val, Y_val_t).item()

        return float(val_loss)

    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        self._train_structure()
        
        self._knockoffs = TimeSeriesKnockoffGenerator.generate(self._data)
        
        # Only use the 20% validation/test data for causality testing
        split_idx = int(self.T * 0.8)
        test_data = self._data[split_idx:]
        test_u = self.u[split_idx:] if self.u is not None else None
        test_knockoffs = self._knockoffs[split_idx:]
        
        # Get baseline (observational) sequences and their true targets
        if self.use_nonstationarity_info:
            X_obs, U_obs, Y_true = self._create_windows(test_data, test_u)
            U_obs_t = torch.FloatTensor(U_obs).to(self.device)
        else:
            X_obs, Y_true = self._create_windows(test_data)
            U_obs_t = None
            
        # Knockoffs only need the main data windows
        if self.use_nonstationarity_info:
            X_knockoffs_obs, _, _ = self._create_windows(test_knockoffs, test_u)
        else:
            X_knockoffs_obs, _ = self._create_windows(test_knockoffs)
        
        self.model.eval()
        with torch.no_grad():
            mu_obs, _ = self.model(torch.FloatTensor(X_obs).to(self.device), U_obs_t)
            Y_pred_obs = mu_obs.cpu().numpy()

        causal_graph = {i: [] for i in range(self.G)}

        for i in range(self.G): 
            cols_i = list(self._groups[i])
            
            for lag in range(1, self.max_lag + 1):
                X_interv = X_obs.copy()
                time_idx = self.max_lag - lag
                
                # Intervene on the specific lag with knockoff data
                X_interv[:, time_idx, cols_i] = X_knockoffs_obs[:, time_idx, cols_i]
                
                with torch.no_grad():
                    mu_interv, _ = self.model(torch.FloatTensor(X_interv).to(self.device), U_obs_t)
                    Y_pred_interv = mu_interv.cpu().numpy()
                    
                for j in range(self.G):  
                    R_j = self._compute_residuals(Y_true, Y_pred_obs, j)
                    R_j_tilde = self._compute_residuals(Y_true, Y_pred_interv, j)
                    
                    max_samples = min(400, len(R_j))
                    idx = np.random.choice(len(R_j), max_samples, replace=False)
                    
                    stat, p_val = ks_2samp(R_j[idx], R_j_tilde[idx])
                    
                    if p_val < self.alpha:
                        causal_graph[j].append((i, -lag))

        return causal_graph