import logging

import numpy as np
import networkx as nx
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import scipy.stats as stats


def check_stationarity(X, significance=0.05):
    """
    Check covariance stationarity for each series in X using the Augmented Dickey-Fuller test.
    Returns a list of booleans indicating whether each series is stationary.
    """
    d = X.shape[1]
    stationary = []
    for i in range(d):
        result = adfuller(X[:, i])
        p_value = result[1]
        stationary.append(p_value < significance)
    return stationary

def create_lag_matrix(X, lags):
    """
    Construct a lagged design matrix from the time series data.
    For each time t (starting at t=lags), the row contains the lagged values
    [X[t-1], X[t-2], ..., X[t-lags]] flattened into a single vector.
    """
    T, d = X.shape
    rows = []
    for t in range(lags, T):
        row = []
        for lag in range(1, lags + 1):
            row.extend(X[t - lag])
        rows.append(row)
    return np.array(rows)


def insert_top_k_edges(G, p_values, max_summarized_crosslinks_density):
    """
    Insert in G the top k% edges with the lowest p-values.
    """
    # We force density = keeping_number / (keeping_number + n)
    #   Because there are n "autolinks"
    keeping_number = int( max_summarized_crosslinks_density * len(G.nodes()) / (1 - max_summarized_crosslinks_density) )
    
    # Sort the p-values in ascending order
    sorted_p_values = sorted(p_values.items(), key=lambda x: x[1])
    
    # Insert the top k% edges
    for (p, q), p_values in sorted_p_values[:keeping_number]:
        G.add_edge(p, q)


def summarized_causality_multivariate_granger(X, max_lag=5, max_summarized_crosslinks_density=0.2,
                                   alpha=0.025):
    """
    Implements the multivariate Granger causality algorithm.
    
    [Geweke, 1982; Chen et al., 2004; Barrett et al.,2010]
    
    Args:
      X      : np.array of shape (T, d) containing the time series data.
      maxlag : Maximum lag to consider for lag order selection.
      max_summarized_crosslinks_density: Maximum fraction of edges that we should supose are going to be cross-links.
      alpha  : Significance level for the F-test - if an edge reaches a lower density, it isn't considered.
      
    Returns:
      G      : A networkx.DiGraph where an edge p -> q indicates that series p
               Granger-causes series q.
    """
    # Check stationarity for each series
    stationarity = check_stationarity(X)
    if not all(stationarity):
        logging.warning("Warning: Some series may not be stationary.")

    # Use VAR to select optimal lag (using AIC)
    var_model = VAR(X)
    order_selection = var_model.select_order(maxlags=max_lag)
    # Use the AIC-selected order; if not found, default to lag 1.
    optimal_lag = order_selection.selected_orders.get('aic', max_lag)
    # Avoid having a too low optimal lag
    if optimal_lag < 1:
        optimal_lag = max(1, max_lag//5)
    # Avoid having a too high optimal lag
    optimal_lag = min(optimal_lag, max_lag//2)

    # Construct the lagged predictor matrix (all variables, all lags)
    X_lag = create_lag_matrix(X, optimal_lag)
    n_obs = X_lag.shape[0]

    # Initialize an empty directed graph with nodes for each series
    G = nx.DiGraph()
    d = X.shape[1]
    G.add_nodes_from(range(d))

    # Generate a dictionary with p-values of each predictor variable
    p_values = {}
    # Loop over each target variable q
    for q in range(d):
        # Response: values of series q, starting from time optimal_lag
        y = X[optimal_lag:, q]

        # Full model: use all lagged predictors (with constant)
        X_full = sm.add_constant(X_lag)
        full_model = sm.OLS(y, X_full).fit()
        RSS_full = np.sum(full_model.resid ** 2)
        df_full = X_full.shape[1]  # number of parameters in full model

        # For each candidate predictor variable p (exclude self)
        for p in range(d):
            if p == q:
                continue

            # Identify columns corresponding to variable p across all lags.
            # The design matrix is organized as:
            # [lag1_var0, lag1_var1, ..., lag1_var(d-1),
            #  lag2_var0, ..., lag_optimalLag_var(d-1)]
            cols_to_remove = [lag * d + p for lag in range(optimal_lag)]
            
            # Build restricted design matrix (drop all lags of variable p)
            X_restricted = np.delete(X_lag, cols_to_remove, axis=1)
            X_restricted = sm.add_constant(X_restricted)
            restricted_model = sm.OLS(y, X_restricted).fit()
            RSS_restricted = np.sum(restricted_model.resid ** 2)
            df_restricted = X_restricted.shape[1]

            # Degrees of freedom difference is number of parameters dropped
            df_diff = df_full - df_restricted

            # Compute the F-statistic:
            F_stat = ((RSS_restricted - RSS_full) / df_diff) / (RSS_full / (n_obs - df_full))
            p_value = 1 - stats.f.cdf(F_stat, df_diff, n_obs - df_full)

            # If p-value is significant, add edge from p to q
            if p_value < alpha:
                p_values[p, q] = p_value
        
        insert_top_k_edges(G, p_values, max_summarized_crosslinks_density)
        
    return G



def summarized_causality_univariate_granger(X, max_lag=5, max_summarized_crosslinks_density=0.2,
                                            alpha=0.025):
    """
    Implements the univariate Granger causality algorithm.
    
    For each pair (p, q) with p != q, it tests whether the lagged values of series p 
    help to predict series q beyond using only the lagged values of series q.
    
    Args:
      X      : np.array of shape (T, d) containing the time series data.
      max_lag: Maximum lag to consider for lag order selection.
      max_summarized_crosslinks_density: Maximum fraction of edges that we should supose are going to be cross-links.
      alpha  : Significance level for the F-test.
      
    Returns:
      G      : A networkx.DiGraph where an edge p -> q indicates that series p
               Granger-causes series q in the univariate sense.
    """
    # Check stationarity for each series (assumed to be defined elsewhere)
    stationarity = check_stationarity(X)
    if not all(stationarity):
        logging.warning("Warning: Some series may not be stationary.")
    
    d = X.shape[1]
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    p_values = {}
    
    # Loop over each target series q
    for q in range(d):
        series_q = X[:, q]
        # Determine optimal lag for series q using univariate AR model selection
        optimal_lag = select_order_univariate(series_q, max_lag)
        if optimal_lag < 1:
            optimal_lag = 1
        
        # Create lagged matrix for series q (for the restricted model)
        # (Assumes create_lag_matrix returns a matrix of lagged predictors)
        X_q_lag = create_lag_matrix(series_q.reshape(-1, 1), optimal_lag)
        y = series_q[optimal_lag:]  # Adjust for the lags
        
        # Restricted model: only lagged values of q
        X_restricted = sm.add_constant(X_q_lag)
        restricted_model = sm.OLS(y, X_restricted).fit()
        RSS_restricted = np.sum(restricted_model.resid ** 2)
        df_restricted = X_restricted.shape[1]
        n_obs = len(y)
        
        # Loop over each candidate predictor series p (excluding q itself)
        for p in range(d):
            if p == q:
                continue
            series_p = X[:, p]
            # Create lag matrix for series p
            X_p_lag = create_lag_matrix(series_p.reshape(-1, 1), optimal_lag)
            
            # Full model: include both lagged values of q and lagged values of p
            X_full = np.hstack([X_q_lag, X_p_lag])
            X_full = sm.add_constant(X_full)
            full_model = sm.OLS(y, X_full).fit()
            RSS_full = np.sum(full_model.resid ** 2)
            df_full = X_full.shape[1]
            
            # Degrees of freedom: number of parameters dropped (lags of p)
            df_diff = optimal_lag
            
            # Compute the F-statistic for the joint hypothesis that lags of p add no predictive power
            F_stat = ((RSS_restricted - RSS_full) / df_diff) / (RSS_full / (n_obs - df_full))
            p_value = 1 - stats.f.cdf(F_stat, df_diff, n_obs - df_full)
            
            # If significant, add a directed edge from p to q
            if p_value < alpha:
                p_values[p, q] = p_value
        
        insert_top_k_edges(G, p_values, max_summarized_crosslinks_density)
    
    return G


def select_order_univariate(series, max_lag):
    """
    Selects the optimal lag order for a univariate time series using AIC.
    
    Args:
      series  : np.array of shape (T,) containing the time series data.
      max_lag : Maximum lag order to consider.
      
    Returns:
      optimal_lag : The lag order that minimizes the AIC.
    """
    from statsmodels.tsa.ar_model import AutoReg
    aic_vals = []
    for lag in range(1, max_lag + 1):
        try:
            model = AutoReg(series, lags=lag, old_names=False).fit()
            aic_vals.append(model.aic)
        except Exception as e:
            aic_vals.append(np.inf)
    optimal_lag = np.argmin(aic_vals) + 1  # +1 because lag indices start at 1
    return optimal_lag



def summarized_causality_ind_test(X, max_lag, max_summarized_crosslinks_density, alpha):
    '''
    Obtain the summarized causality graph, using independence test with the whole lags of each variable.
    '''
    




# Example usage:
if __name__ == "__main__":
    # Generate synthetic data: 200 time points, 4 variables.
    np.random.seed(42)
    T = 200
    d = 3
    X = np.random.randn(T, d)
    X_lagged = np.roll(X, 2, axis=0) + np.random.randn(T, d) * 0.1
    X = np.append(X, X_lagged, axis=1)
    logging.info("Time series data shape: %s", X.shape)
    logging.info("Time series data:\n%s", X)

    # Run the multivariate Granger causality procedure
    G = summarized_causality_multivariate_granger(X, max_lag=5, alpha=0.05)
    logging.info("Edges in the Granger causality graph (p -> q):")
    logging.info("Edges: %s", list(G.edges()))
