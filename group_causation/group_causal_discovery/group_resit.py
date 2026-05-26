import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Union, Optional
from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.independence_tests import HSIC_Test


# ---------------------------------------------------------------------------
# MLP Regressors (Standard & Spatio-Temporal MURGS)
# ---------------------------------------------------------------------------
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim: int, u_dim: int, output_dim: int, hidden_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + u_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, u=None):
        if u is not None:
            x = torch.cat([x, u], dim=-1)
        return self.net(x)


class GroupRegressor:
    """Standard Regressor used for Phase I residual computation."""
    def __init__(self, epochs=200, batch_size=200, lr=0.01, hidden_dim=100):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dim = hidden_dim

    def fit(self, X: np.ndarray, Y: np.ndarray, U: Optional[np.ndarray] = None):
        input_dim = X.shape[1]
        u_dim = U.shape[1] if U is not None else 0
        output_dim = Y.shape[1]
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiOutputMLP(input_dim, u_dim, output_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        if U is not None:
            dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(U), torch.FloatTensor(Y))
        else:
            dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
            
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for batch in loader:
                if U is not None:
                    batch_x, batch_u, batch_y = batch
                    batch_u = batch_u.to(self.device)
                else:
                    batch_x, batch_y = batch
                    batch_u = None
                    
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                preds = self.model(batch_x, batch_u)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            U_t = torch.FloatTensor(U).to(self.device) if U is not None else None
            preds = self.model(X_t, U_t)
        return preds.cpu().numpy()


class SpatioTemporalMURGSRegressor(GroupRegressor):
    """
    Regressor with L2,1 Group Lasso Penalty on the input layer weights 
    to implement the Temporal-MURGS pruning for Phase II.
    """
    def __init__(self, epochs=200, batch_size=200, lr=0.01, hidden_dim=100, lambda_reg=0.01):
        super().__init__(epochs, batch_size, lr, hidden_dim)
        self.lambda_reg = lambda_reg

    def fit(self, X: np.ndarray, Y: np.ndarray, group_dims: list[int], U: Optional[np.ndarray] = None):
        input_dim = X.shape[1]
        u_dim = U.shape[1] if U is not None else 0
        output_dim = Y.shape[1]
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiOutputMLP(input_dim, u_dim, output_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        if U is not None:
            dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(U), torch.FloatTensor(Y))
        else:
            dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
            
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for batch in loader:
                if U is not None:
                    batch_x, batch_u, batch_y = batch
                    batch_u = batch_u.to(self.device)
                else:
                    batch_x, batch_y = batch
                    batch_u = None
                    
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                preds = self.model(batch_x, batch_u)
                mse_loss = criterion(preds, batch_y)
                
                # Apply Spatio-Temporal Group Lasso (MURGS Penalty)
                reg_loss = 0.0
                start_idx = 0
                W_in = self.model.net[0].weight
                
                # Note: Because group_dims sum to input_dim, this loop naturally
                # skips penalizing the last u_dim columns associated with U.
                for g_dim in group_dims:
                    end_idx = start_idx + g_dim
                    W_group = W_in[:, start_idx:end_idx] # type: ignore
                    reg_loss += math.sqrt(g_dim) * torch.norm(W_group, p='fro')
                    start_idx = end_idx
                
                loss = mse_loss + self.lambda_reg * reg_loss
                loss.backward()
                optimizer.step()

    def get_group_norms(self, group_dims: list[int]) -> list[float]:
        self.model.eval()
        norms = []
        start_idx = 0
        with torch.no_grad():
            W_in = self.model.net[0].weight.cpu()
            for g_dim in group_dims:
                end_idx = start_idx + g_dim
                W_group = W_in[:, start_idx:end_idx] # type: ignore
                norms.append(float(torch.norm(W_group, p='fro')))
                start_idx = end_idx
        return norms


# ---------------------------------------------------------------------------
# Time-Series GroupRESIT-MURGS Algorithm
# ---------------------------------------------------------------------------
class GroupRESITTimeSeriesCausalDiscovery(GroupCausalDiscovery):
    '''
    Time-Series adaptation of the GroupRESIT Algorithm.
    Phase I: HSIC-based Sink Node identification for contemporaneous order.
    Phase II: Spatio-Temporal MURGS pruning via Group-Lasso Neural Networks.
    '''
    def __init__(self, data: np.ndarray, groups: Union[list[set[int]], None] = None,
                 standarize: bool=True, non_stationarity_info: Union[dict, None] = None, 
                 use_nonstationarity_info: bool = False,
                 verbose: int = 0, **kwargs):
                 
        super().__init__(data, groups, standarize, **kwargs)
        
        self.epochs = self.extra_args.get("epochs", 200)
        self.hidden_dim = self.extra_args.get("hidden_dim", 100)
        self.max_lag = self.extra_args.get("max_lag", 1)
        self.min_lag = self.extra_args.get("min_lag", 1) 
        
        self.lambda_reg = self.extra_args.get("lambda_reg", 0.05)
        self.pruning_threshold = self.extra_args.get("pruning_threshold", 1e-3)
        self._verbose = verbose
        
        self.T = self._data.shape[0]
        self.G = len(self._groups)
        
        if self.T <= self.max_lag:
            raise ValueError("Time series length T must be strictly greater than max_lag.")
        if self.min_lag > self.max_lag:
            raise ValueError("min_lag cannot be strictly greater than max_lag.")
            
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
        
        self._causal_order = [] 
        self._pa = {}           

    def _get_data_and_dims_for_vars(self, vars_list: list[tuple[int, int]], data: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[int]]:
        """
        Constructs a flat 2D array of specific groups at specific lags,
        and returns the feature dimensions of each group block for the MURGS penalty.
        """
        current_data = self._data if data is None else data

        if not vars_list:
            return np.empty((current_data.shape[0] - self.max_lag, 0)), []
            
        blocks = []
        dims = []
        for g, l in vars_list:
            cols = list(self._groups[g])
            start_idx = self.max_lag - l
            end_idx = current_data.shape[0] - l
            blocks.append(current_data[start_idx:end_idx, cols])
            dims.append(len(cols))
            
        return np.concatenate(blocks, axis=1), dims

    def score_validation_mse(self, validation_data: np.ndarray) -> float:
        """Scores a simple validation proxy based on the MURGS regression loss on a held-out split."""
        start_lag = max(1, self.min_lag)
        potential_parents = [(g, l) for g in range(self.G) for l in range(start_lag, self.max_lag + 1)]

        if self.min_lag == 0:
            potential_parents = [(g, 0) for g in range(self.G)] + potential_parents

        if self._data.shape[0] <= self.max_lag or validation_data.shape[0] <= self.max_lag:
            return float('inf')

        X_train, group_dims = self._get_data_and_dims_for_vars(potential_parents, data=self._data)
        X_val, _ = self._get_data_and_dims_for_vars(potential_parents, data=validation_data)

        if X_train.size == 0 or X_val.size == 0:
            return float('inf')

        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std

        group_scores = []
        for k in range(self.G):
            Y_train, _ = self._get_data_and_dims_for_vars([(k, 0)], data=self._data)
            Y_val, _ = self._get_data_and_dims_for_vars([(k, 0)], data=validation_data)

            Y_mean = Y_train.mean(axis=0)
            Y_std = Y_train.std(axis=0) + 1e-8
            Y_train_std = (Y_train - Y_mean) / Y_std
            Y_val_std = (Y_val - Y_mean) / Y_std

            murgs_model = SpatioTemporalMURGSRegressor(
                epochs=self.epochs,
                hidden_dim=self.hidden_dim,
                lambda_reg=self.lambda_reg,
            )
            murgs_model.fit(X_train, Y_train_std, group_dims)
            Y_pred = murgs_model.predict(X_val)
            group_scores.append(float(np.mean((Y_pred - Y_val_std) ** 2)))

        return float(np.mean(group_scores))

    def _phase_1_causal_order(self):
        """Phase I: Infer the causal order among contemporary variables (lag 0)."""
        if self.min_lag > 0:
            self._causal_order = list(range(self.G))
            return

        S = list(range(self.G))
        pi_contemp = []
        
        start_lag = max(1, self.min_lag)
        past_vars = [(g, l) for g in range(self.G) for l in range(start_lag, self.max_lag + 1)]
        
        # FIXED: Checking directly if `self.u is not None` to prevent slicing errors
        U_sliced = self.u[self.max_lag:] if self.u is not None else None

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

                regressor = GroupRegressor(epochs=self.epochs, hidden_dim=self.hidden_dim)
                regressor.fit(X, Y, U_sliced)
                Y_pred = regressor.predict(X, U_sliced)
                
                residuals = Y - Y_pred
                residuals = (residuals - residuals.mean(axis=0)) / (residuals.std(axis=0) + 1e-8)
                X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
                
                # HSIC Test on structurally independent variables vs residuals
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

        U_sliced = self.u[self.max_lag:] if self.u is not None else None

        for i, k in enumerate(self._causal_order):
            contemp_preds = [(p, 0) for p in self._causal_order[:i]] if self.min_lag == 0 else []
            potential_parents = contemp_preds + past_vars
            
            if not potential_parents:
                pa[k] = []
                continue

            X_pot_parents, group_dims = self._get_data_and_dims_for_vars(potential_parents)
            Y, _ = self._get_data_and_dims_for_vars([(k, 0)])
            
            X_pot_parents = (X_pot_parents - X_pot_parents.mean(axis=0)) / (X_pot_parents.std(axis=0) + 1e-8)
            Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)

            murgs_model = SpatioTemporalMURGSRegressor(
                epochs=self.epochs, 
                hidden_dim=self.hidden_dim, 
                lambda_reg=self.lambda_reg
            )
            murgs_model.fit(X_pot_parents, Y, group_dims, U_sliced)
            
            norms = murgs_model.get_group_norms(group_dims)
            
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