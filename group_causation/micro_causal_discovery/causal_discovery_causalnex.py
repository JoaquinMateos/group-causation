import pandas as pd
import numpy as np

from group_causation.micro_causal_discovery.micro_causal_discovery_base import MicroCausalDiscovery


class DynotearsWrapper(MicroCausalDiscovery):
    '''
    Wrapper for DYNOTEARS algorithm
    
    Args:
        data : np.array with the data, shape (n_samples, n_features)
        max_lag : maximum lag to consider
    '''
    def __init__(self, data: np.ndarray, min_lag: int, max_lag: int, **kwargs):
        super().__init__(data, **kwargs)
        
        self.df = pd.DataFrame(self._data, columns=range(self._data.shape[1]))
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.kwargs = kwargs
        
    def extract_parents(self) -> dict[int, list[int]]:
        '''
        Returns the parents dict
        
        Args:
            data : np.array with the data, shape (n_samples, n_features)
        '''
        predicted_parents = from_pandas_dynamic(time_series=self.df,
                                                        p=self.max_lag,
                                                        **self.kwargs)
        
        if self.min_lag > 0:
            predicted_parents = {
                k: [(v, lag) for v, lag in v_lags if lag >= self.min_lag]
                for k, v_lags in predicted_parents.items()
            }
            
        
        return predicted_parents
    




# CODE IN THE REST OF THIS FILE IS ADAPTED FROM THE CAUSALNEX LIBRARY
# Copyright 2019-2020 QuantumBlack Visual Analytics Limited
# SPDX-License-Identifier: Apache-2.0
# THE ADDAPTATIONS ARE ONLY RELATED TO THE INTERFACE OF THE CLASS AND NOT TO THE FUNCTIONALITY




"""
Tools to learn a Dynamic Bayesian Network which describe the conditional dependencies between variables in a time-series
dataset.
"""
import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt

def from_pandas_dynamic(
        time_series: Union[pd.DataFrame, List[pd.DataFrame]],
        p: int,
        lambda_w: float = 0.1,
        lambda_a: float = 0.1,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        w_threshold: float = 0.0,
        tabu_edges: List[Tuple[int, int, int]] = None,
        tabu_parent_nodes: List[int] = None,
        tabu_child_nodes: List[int] = None,
    ) -> dict[int, list[tuple[int, int]]]:
    """
    Learn the graph structure of a Dynamic Bayesian Network describing conditional dependencies between variables in
    data. The input data is a time series or a list of realisations of a same time series.
    The optimisation is to minimise a score function F(W, A) over the graph's contemporaneous (intra-slice) weighted
    adjacency matrix, W, and lagged (inter-slice) weighted adjacency matrix, A, subject to the a constraint function
    h(W), where h_value(W) == 0 characterises an acyclic graph. h(W) > 0 is a continuous, differentiable function that
    encapsulated how acyclic the graph is (less = more acyclic).

    Based on "DYNOTEARS: Structure Learning from Time-Series Data".
    
    .. code-block:: bibtex
    
        https://arxiv.org/abs/2002.00498
        @inproceedings{pamfil2020dynotears,
            title={DYNOTEARS: Structure Learning from Time-Series Data},
            author={Pamfil, Roxana and Sriwattanaworachai, Nisara and Desai, Shaan and Pilgerstorfer,
            Philip and Georgatzis, Konstantinos and Beaumont, Paul and Aragam, Bryon},
            booktitle={International Conference on Artificial Intelligence and Statistics},
            pages={1595--1605},
            year={2020},
        }
    
    Args:
        time_series: pd.DataFrame or List of pd.DataFrame instances.
        If a list is provided each element of the list being an realisation of a time series (i.e. time series governed
        by the same processes)
        The columns of the data frame represent the variables in the model, and the *index represents the time index*.
        Successive events, therefore, must be indexed with one integer of difference between them too.
        p: Number of past interactions we allow the model to create. The state of a variable at time `t` is affected by
        past variables up to a `t-p`, as well as by other variables at `t`.
        lambda_w: parameter for l1 regularisation of intra-slice edges
        lambda_a: parameter for l1 regularisation of inter-slice edges
        max_iter: max number of dual ascent steps during optimisation.
        h_tol: exit if h(W) < h_tol (as opposed to strict definition of 0).
        w_threshold: fixed threshold for absolute edge weights.
        tabu_edges: list of edges(lag, from, to) not to be included in the graph. `lag == 0` implies that the edge is
        forbidden in the INTRA graph (W), while lag > 0 implies an INTER-slice weight equal zero.
        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
        tabu_child_nodes: list of nodes banned from being a child of any other nodes.

    Returns:
        Dictionary with the parents of each node. The dictionary is in the format {son: [(parent, lag), ...]} where son
        is the index of the child node and parent is the index of the parent node. The lag is the time lag of the parent
        node with respect to the child node.
    """
    time_series = [time_series] if not isinstance(time_series, list) else time_series

    X, Xlags = convert_to_dynotears_format(time_series, p)

    col_idx = {c: i for i, c in enumerate(time_series[0].columns)}

    if tabu_edges:
        tabu_edges = [(lag, col_idx[u], col_idx[v]) for lag, u, v in tabu_edges]
    if tabu_parent_nodes:
        tabu_parent_nodes = [col_idx[n] for n in tabu_parent_nodes]
    if tabu_child_nodes:
        tabu_child_nodes = [col_idx[n] for n in tabu_child_nodes]

    g = from_numpy_dynamic(
        X,
        Xlags,
        lambda_w,
        lambda_a,
        max_iter,
        h_tol,
        w_threshold,
        tabu_edges,
        tabu_parent_nodes,
        tabu_child_nodes,
    )
    return g


def from_numpy_dynamic(
        X: np.ndarray,
        Xlags: np.ndarray,
        lambda_w: float = 0.1,
        lambda_a: float = 0.1,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        w_threshold: float = 0.0,
        tabu_edges: List[Tuple[int, int, int]] = None,
        tabu_parent_nodes: List[int] = None,
        tabu_child_nodes: List[int] = None,
    ) -> dict[int, list[tuple[int, int]]]:
    """
    Learn the graph structure of a Dynamic Bayesian Network describing conditional dependencies between variables in
    data. The input data is time series data present in numpy arrays X and Xlags.

    The optimisation is to minimise a score function F(W, A) over the graph's contemporaneous (intra-slice) weighted
    adjacency matrix, W, and lagged (inter-slice) weighted adjacency matrix, A, subject to the a constraint function
    h(W), where h_value(W) == 0 characterises an acyclic graph. h(W) > 0 is a continuous, differentiable function that
    encapsulated how acyclic the graph is (less = more acyclic).

    Based on "DYNOTEARS: Structure Learning from Time-Series Data".
    https://arxiv.org/abs/2002.00498
    
    .. code-block:: bibtex
    
        @inproceedings{pamfil2020dynotears,
            title={DYNOTEARS: Structure Learning from Time-Series Data},
            author={Pamfil, Roxana and Sriwattanaworachai, Nisara and Desai, Shaan and Pilgerstorfer,
            Philip and Georgatzis, Konstantinos and Beaumont, Paul and Aragam, Bryon},
            booktitle={International Conference on Artificial Intelligence and Statistics},
            pages={1595--1605},
            year={2020},
        }

    Args:
        X (np.ndarray): 2d input data, axis=1 is data columns, axis=0 is data rows. Each column represents one variable,
            and each row represents x(m,t) i.e. the mth time series at time t.
        Xlags (np.ndarray): shifted data of X with lag orders stacking horizontally. Xlags=[shift(X,1)|...|shift(X,p)]
        lambda_w (float): l1 regularization parameter of intra-weights W
        lambda_a (float): l1 regularization parameter of inter-weights A
        max_iter: max number of dual ascent steps during optimisation
        h_tol (float): exit if h(W) < h_tol (as opposed to strict definition of 0)
        w_threshold: fixed threshold for absolute edge weights.
        tabu_edges: list of edges(lag, from, to) not to be included in the graph. `lag == 0` implies that the edge is
        forbidden in the INTRA graph (W), while lag > 0 implies an INTER weight equal zero.
        tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
        tabu_child_nodes: list of nodes banned from being a child of any other nodes.
    
    Returns:
        Dictionary with the parents of each node. The dictionary is in the format {son: [(parent, lag), ...]} where son
        is the index of the child node and parent is the index of the parent node. The lag is the time lag of the parent
        node with respect to the child node.

    Raises:
        ValueError: If X or Xlags does not contain data, or dimensions of X and Xlags do not conform
    """
    _, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    bnds_w = 2 * [
        (0, 0)
        if i == j
        else (0, 0)
        if tabu_edges is not None and (0, i, j) in tabu_edges
        else (0, 0)
        if tabu_parent_nodes is not None and i in tabu_parent_nodes
        else (0, 0)
        if tabu_child_nodes is not None and j in tabu_child_nodes
        else (0, None)
        for i in range(d_vars)
        for j in range(d_vars)
    ]

    bnds_a = []
    for k in range(1, p_orders + 1):
        bnds_a.extend(
            2
            * [
                (0, 0)
                if tabu_edges is not None and (k, i, j) in tabu_edges
                else (0, 0)
                if tabu_parent_nodes is not None and i in tabu_parent_nodes
                else (0, 0)
                if tabu_child_nodes is not None and j in tabu_child_nodes
                else (0, None)
                for i in range(d_vars)
                for j in range(d_vars)
            ]
        )

    bnds = bnds_w + bnds_a
    w_est, a_est = _learn_dynamic_structure(
        X, Xlags, bnds, lambda_w, lambda_a, max_iter, h_tol
    )

    w_est[np.abs(w_est) < w_threshold] = 0
    a_est[np.abs(a_est) < w_threshold] = 0
    parents = _matrices_to_parents(w_est, a_est)
    
    return parents


def _matrices_to_parents(
    w_est: np.ndarray, a_est: np.ndarray
) -> dict[int, list[tuple[int, int]]]:
    """
    Converts the matrices output by dynotears (W and A) into a dictionary of parents.
    The dictionary is in the format {son: [(parent, lag), ...]} where son is the index of the child node and
    parent is the index of the parent node. The lag is the time lag of the parent node with respect to the child node.
    Args:
        w_est: Intra-slice weight matrix
        a_est: Inter-slice matrix

    Returns:
        Dictionary with the parents of each node

    """
    n_vars = w_est.shape[0]
    parents_contemporaneous = {son: [(parent, 0)
                                    for parent in range(n_vars)
                                    if w_est[parent, son] != 0] # w_est[father, son] is the weight
                                for son in range(n_vars)}
    
    parents_lagged = {son: [(parent, -1-lag)
                                    for parent in range(n_vars)
                                    for lag in range(a_est.shape[0]//n_vars)
                                    if a_est[parent+lag*n_vars, son] != 0] # a_est[father, son] is the weight
                                for son in range(n_vars)}
    
    parents = {son: parents_contemporaneous[son] + parents_lagged[son]
                for son in parents_contemporaneous.keys()}
    
    return parents


def _reshape_wa(
    wa_vec: np.ndarray, d_vars: int, p_orders: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function for `_learn_dynamic_structure`. Transform adjacency vector to matrix form

    Args:
        wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights
        d_vars (int): number of variables in the model
        p_orders (int): number of past indexes we to use
    Returns:
        intra- and inter-slice adjacency matrices
    """

    w_tilde = wa_vec.reshape([2 * (p_orders + 1) * d_vars, d_vars])
    w_plus = w_tilde[:d_vars, :]
    w_minus = w_tilde[d_vars : 2 * d_vars, :]
    w_mat = w_plus - w_minus
    a_plus = (
        w_tilde[2 * d_vars :]
        .reshape(2 * p_orders, d_vars**2)[::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_minus = (
        w_tilde[2 * d_vars :]
        .reshape(2 * p_orders, d_vars**2)[1::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_mat = a_plus - a_minus
    return w_mat, a_mat


def _learn_dynamic_structure(
    X: np.ndarray,
    Xlags: np.ndarray,
    bnds: List[Tuple[float, float]],
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learn the graph structure of a Dynamic Bayesian Network describing conditional dependencies between data variables.

    The optimisation is to minimise a score function F(W, A) over the graph's contemporaneous (intra-slice) weighted
    adjacency matrix, W, and lagged (inter-slice) weighted adjacency matrix, A, subject to the a constraint function
    h(W), where h_value(W) == 0 characterises an acyclic graph. h(W) > 0 is a continuous, differentiable function that
    encapsulated how acyclic the graph is (less = more acyclic).

    Based on "DYNOTEARS: Structure Learning from Time-Series Data".
    https://arxiv.org/abs/2002.00498
    @inproceedings{pamfil2020dynotears,
        title={DYNOTEARS: Structure Learning from Time-Series Data},
        author={Pamfil, Roxana and Sriwattanaworachai, Nisara and Desai, Shaan and Pilgerstorfer,
        Philip and Georgatzis, Konstantinos and Beaumont, Paul and Aragam, Bryon},
        booktitle={International Conference on Artificial Intelligence and Statistics},
        pages={1595--1605},
        year={2020}year={2020},
    }

    Args:
        X (np.ndarray): 2d input data, axis=1 is data columns, axis=0 is data rows. Each column represents one variable,
        and each row represents x(m,t) i.e. the mth time series at time t.
        Xlags (np.ndarray): shifted data of X with lag orders stacking horizontally. Xlags=[shift(X,1)|...|shift(X,p)]
        bnds: Box constraints of L-BFGS-B to ban self-loops in W, enforce non-negativity of w_plus, w_minus, a_plus,
        a_minus, and help with stationarity in A
        lambda_w (float): l1 regularization parameter of intra-weights W
        lambda_a (float): l1 regularization parameter of inter-weights A
        max_iter (int): max number of dual ascent steps during optimisation
        h_tol (float): exit if h(W) < h_tol (as opposed to strict definition of 0)

    Returns:
        W (np.ndarray): d x d estimated weighted adjacency matrix of intra slices
        A (np.ndarray): d x pd estimated weighted adjacency matrix of inter slices

    Raises:
        ValueError: If X or Xlags does not contain data, or dimensions of X and Xlags do not conform
    """
    if X.size == 0:
        raise ValueError("Input data X is empty, cannot learn any structure")
    if Xlags.size == 0:
        raise ValueError("Input data Xlags is empty, cannot learn any structure")
    if X.shape[0] != Xlags.shape[0]:
        raise ValueError("Input data X and Xlags must have the same number of rows")
    if Xlags.shape[1] % X.shape[1] != 0:
        raise ValueError(
            "Number of columns of Xlags must be a multiple of number of columns of X"
        )

    n, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    def _h(wa_vec: np.ndarray) -> float:
        """
        Constraint function of the dynotears

        Args:
            wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights

        Returns:
            float: DAGness of the intra-slice adjacency matrix W (0 == DAG, >0 == cyclic)
        """

        _w_mat, _ = _reshape_wa(wa_vec, d_vars, p_orders)
        return np.trace(slin.expm(_w_mat * _w_mat)) - d_vars

    def _func(wa_vec: np.ndarray) -> float:
        """
        Objective function that the dynotears tries to minimise

        Args:
            wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights

        Returns:
            float: objective
        """

        _w_mat, _a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
        loss = (
            0.5
            / n
            * np.square(
                np.linalg.norm(
                    X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat), "fro"
                )
            )
        )
        _h_value = _h(wa_vec)
        l1_penalty = lambda_w * (wa_vec[: 2 * d_vars**2].sum()) + lambda_a * (
            wa_vec[2 * d_vars**2 :].sum()
        )
        return loss + 0.5 * rho * _h_value * _h_value + alpha * _h_value + l1_penalty

    def _grad(wa_vec: np.ndarray) -> np.ndarray:
        """
        Gradient function used to compute next step in dynotears

        Args:
            wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights

        Returns:
            gradient vector
        """

        _w_mat, _a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
        e_mat = slin.expm(_w_mat * _w_mat)
        loss_grad_w = (
            -1.0
            / n
            * (X.T.dot(X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat)))
        )
        obj_grad_w = (
            loss_grad_w
            + (rho * (np.trace(e_mat) - d_vars) + alpha) * e_mat.T * _w_mat * 2
        )
        obj_grad_a = (
            -1.0
            / n
            * (Xlags.T.dot(X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat)))
        )

        grad_vec_w = np.append(
            obj_grad_w, -obj_grad_w, axis=0
        ).flatten() + lambda_w * np.ones(2 * d_vars**2)
        grad_vec_a = obj_grad_a.reshape(p_orders, d_vars**2)
        grad_vec_a = np.hstack(
            (grad_vec_a, -grad_vec_a)
        ).flatten() + lambda_a * np.ones(2 * p_orders * d_vars**2)
        return np.append(grad_vec_w, grad_vec_a, axis=0)

    # initialise matrix, weights and constraints
    wa_est = np.zeros(2 * (p_orders + 1) * d_vars**2)
    wa_new = np.zeros(2 * (p_orders + 1) * d_vars**2)
    rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf

    for n_iter in range(max_iter):
        while (rho < 1e20) and (h_new > 0.25 * h_value or h_new == np.inf):
            wa_new = sopt.minimize(
                _func, wa_est, method="L-BFGS-B", jac=_grad, bounds=bnds
            ).x
            h_new = _h(wa_new)
            if h_new > 0.25 * h_value:
                rho *= 10

        wa_est = wa_new
        h_value = h_new
        alpha += rho * h_value
        if h_value <= h_tol:
            break
        if h_value > h_tol and n_iter == max_iter - 1:
            warnings.warn("Failed to converge. Consider increasing max_iter.")
    return _reshape_wa(wa_est, d_vars, p_orders)


def convert_to_dynotears_format(
    time_series: Union[pd.DataFrame, List[pd.DataFrame]], p: int, columns: List[str] = None
) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    """
    Applies transformation to format the dataframe properly
    
    Args:
        time_series: time_series: pd.DataFrame or List of pd.DataFrame instances. Details on `fit` documentation
        p: Number of past interactions we allow the model to create. The state of a variable at time `t` is affected by
            past variables up to a `t-p`, as well as by other variables at `t`.
        columns: list of columns to be used. If None, all columns are used.
    Returns:
        tuple of two numpy.ndarrayy: X and Xlags
                X (np.ndarray): 2d input data, axis=1 is data columns, axis=0 is data rows.
                    Each column represents one variable,
                    and each row represents x(m,t) i.e. the mth time series at time t.
                Xlags (np.ndarray):
                    Shifted data of X with lag orders stacking horizontally. Xlags=[shift(X,1)|...|shift(X,p)]
    """
    time_series = time_series if isinstance(time_series, list) else [time_series]

    if columns is None:
        columns = time_series[0].columns.tolist()
    
    _check_input_from_pandas(time_series, columns)

    time_series = [t[columns] for t in time_series]
    ts_realisations = _cut_dataframes_on_discontinuity_points(time_series)
    X, Xlags = _convert_realisations_into_dynotears_format(
        ts_realisations, p
    )

    return X, Xlags


def _check_input_from_pandas(time_series: List[pd.DataFrame], columns: list[str]):
    """
    Check if the input of function `from_pandas_dynamic` is valid
    
    Args:
        time_series: List of pd.DataFrame instances.
            each element of the list being an realisation of a same time series

    Raises:
        ValueError: if empty list of time_series is provided
        ValueError: if dataframes contain non numeric data
        TypeError: if elements provided are not pandas dataframes
        ValueError: if dataframes contain different columns
        ValueError: if dataframes index is not in increasing order
        TypeError: if dataframes index are not index
    """
    if not time_series:
        raise ValueError(
            "Provided empty list of time_series. At least one DataFrame must be provided"
        )

    df = time_series[0]

    for t in time_series:
        if not isinstance(t, pd.DataFrame):
            raise TypeError(
                "Time series entries must be instances of `pd.DataFrame`"
            )

        non_numeric_cols = t.select_dtypes(exclude="number").columns

        if not non_numeric_cols.empty:
            raise ValueError(
                "All columns must have numeric data. Consider mapping the "
                f"following columns to int: {list(non_numeric_cols)}"
            )

        if (not np.all(df.columns == t.columns)) or (
            not np.all(df.dtypes == t.dtypes)
        ):
            raise ValueError("All inputs must have the same columns and same types")

        if not np.all(t.index == t.index.sort_values()):
            raise ValueError(
                "Index for dataframe must be provided in increasing order"
            )

        if not pd.api.types.is_integer_dtype(t.index):
            raise TypeError("Index must be integers")

        missing_cols = [c for c in columns if c not in t.columns]
        if missing_cols:
            raise ValueError(
                "We should provide all necessary columns in the time series. "
                f"Columns not provided: {missing_cols}"
            )

def _cut_dataframes_on_discontinuity_points(
    time_series: List[pd.DataFrame],
) -> List[np.ndarray]:
    """
    Helper function for `from_pandas_dynamic`
    Receive a list of dataframes. For each dataframe, cut the points of discontinuity as two different dataframes.
    Discontinuities are determined by the indexes.

    For Example:
    If the following is a dataframe:
        index   variable_1  variable_2
        1       X           X
        2       X           X
        3       X           X
        4       X           X
        8       X           X               <- discontinuity point
        9       X           X
        10      X           X

    We cut this dataset in two:

        index   variable_1  variable_2
        1       X           X
        2       X           X
        3       X           X
        4       X           X

        and:
        index   variable_1  variable_2
        8       X           X
        9       X           X
        10      X           X


    Args:
        time_series: list of dataframes representing various realisations of a same time series

    Returns:
        List of np.ndarrays representing the pieces of the input datasets with no discontinuity

    """
    time_series_realisations = []
    for t in time_series:
        cutting_points = np.where(np.diff(t.index) > 1)[0]
        cutting_points = [0] + list(cutting_points + 1) + [len(t)]
        for start, end in zip(cutting_points[:-1], cutting_points[1:]):
            time_series_realisations.append(t.iloc[start:end, :].values)
    return time_series_realisations

def _convert_realisations_into_dynotears_format(
    realisations: List[np.ndarray], p: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of realisations of a time series, convert it to the format received by the dynotears algorithm.
    Each realisation on `realisations` is a realisation of the time series,
    where the time dimension is represented by the rows.
        - The higher the row, the higher the time index
        - The data is complete, meaning that the difference between two time stamps is equal one
    Args:
        realisations: a list of realisations of a time series
        p: the number of lagged columns to create

    Returns:
        X and Y as in the SVAR model and DYNOTEARS paper. I.e. X being representing X(m,t) and Y the concatenated
        differences [X(m,t-1) | X(m,t-2) | ... | X(m,t-p)]
    """
    X = np.concatenate([realisation[p:] for realisation in realisations], axis=0)
    y_lag_list = [
        np.concatenate([realisation[p - i - 1 : -i - 1] for i in range(p)], axis=1)
        for realisation in realisations
    ]
    y_lag = np.concatenate(y_lag_list, axis=0)

    return X, y_lag







if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)

    # Example data
    data = np.random.rand(100, 5)  # 100 samples, 5 features
    max_lag = 2

    # Create an instance of the DynotearsWrapper
    dynotears = DynotearsWrapper(data, max_lag)

    # Extract parents
    predicted_parents = dynotears.extract_parents()
    
    known_parents = {0: [(2, 0), (2, -2), (3, -1), (3, -2), (4, -2)], 1: [(3, 0), (1, -1), (1, -2), (2, -1), (2, -2), (3, -1)], 2: [(0, 0), (4, 0), (1, -1), (2, -2), (4, -2)], 3: [(1, 0), (1, -1), (3, -1)], 4: [(2, 0), (0, -2), (3, -1), (3, -2), (4, -1)]}
    print(f'{predicted_parents=}')
    assert(predicted_parents == known_parents)
    print('Test passed correctly')