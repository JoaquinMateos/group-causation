from typing import Optional, Union, Any
import numpy as np
import logging
from sklearn.linear_model import LinearRegression

import tigramite
import tigramite.data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

# To admit the use of this package's data structures
from causalai.data.time_series import TimeSeriesData

from group_causation.micro_causal_discovery.micro_causal_discovery_base import MicroCausalDiscovery


class PCMCIWrapper(MicroCausalDiscovery):
    '''
    Wrapper for PCMCI algorithm with support for localized independence tests.
    
    Args:
        data: np.array with the data, shape (n_samples, n_features)
        cond_ind_test: string with the name of the conditional independence test 
                       ('shift_based_local_hsic' for given regime shifts,
                        'localized_hsic' to equally divide the series, or standard tests).
        min_lag: minimum lag to consider
        max_lag: maximum lag to consider
        pc_alpha: alpha value for the conditional independence test
        non_stationarity_info: dict specifying regime shifts to guide shift_based_local_hsic
    '''
    def __init__(self, data: np.ndarray, cond_ind_test='parcorr',
                 min_lag=1, max_lag=3, pc_alpha: Optional[float] = None, 
                 non_stationarity_info: Optional[dict[str, Any]] = None, **kwargs):
        super().__init__(data, **kwargs)
        
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.pc_alpha = pc_alpha
        self.extra_args = kwargs
        non_stationarity_info = non_stationarity_info if non_stationarity_info is not None else {}

        # ---------------------------------------------------------
        # Background 'u' Construction (for Localized HSIC)
        # ---------------------------------------------------------
        self.u = None
        if cond_ind_test == 'shift_based_local_hsic' and non_stationarity_info.get('type') == 'regime_shifts':
            affected_vars = non_stationarity_info.get('affected_vars', [])
            if affected_vars:
                first_var = affected_vars[0]
                shifts = non_stationarity_info['shift_details'][first_var]
                
                total_T = shifts[-1]['end']
                u_full = np.zeros(total_T, dtype=int)
                for shift in shifts:
                    u_full[shift['start']:shift['end']] = shift['regime']
                
                T_data = data.shape[0]
                u_aligned = u_full[-T_data:]
                
                num_regimes = non_stationarity_info.get('num_shifts', len(shifts)) + 1
                u_one_hot = np.zeros((T_data, num_regimes))
                u_one_hot[np.arange(T_data), u_aligned] = 1
                self.u = u_one_hot
                
        elif cond_ind_test == 'localized_hsic':
            # Divide the time series into equally spaced chunks
            num_regimes = self.extra_args.get('num_regimes', 5) # Default to 5 chunks
            T_data = data.shape[0]
            chunk_size = int(np.ceil(T_data / num_regimes))
            
            u_aligned = np.repeat(np.arange(num_regimes), chunk_size)[:T_data]
            u_one_hot = np.zeros((T_data, num_regimes))
            u_one_hot[np.arange(T_data), u_aligned] = 1
            self.u = u_one_hot

        # Map to chosen conditional independence test
        if cond_ind_test in ['shift_based_local_hsic', 'localized_hsic']:
            self.cond_ind_test = LocalizedResidualHSIC(self._data, self.u)
        else:
            self.cond_ind_test = {'parcorr': ParCorr(),
                                  'gpdc': GPDC(),
                                  'cmiknn': CMIknn(significance='fixed_thres'), # Very slow
                                 }[cond_ind_test]
        
        # Convert to Tigramite DataFrame format
        dataframe = convert_to_tigramite_dataframe(self._data)
        self.pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=self.cond_ind_test,
            verbosity=0,
        )
    
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        '''
        Returns the parents dict
        '''
        # Temporarily remove custom num_regimes argument so Tigramite doesn't complain
        safe_extra_args = {k: v for k, v in self.extra_args.items() if k != 'num_regimes'}
        
        results = self.pcmci.run_pcmciplus(
            tau_min=self.min_lag, 
            tau_max=self.max_lag,
            pc_alpha=(self.pc_alpha if self.pc_alpha is not None else 0.05), 
            **safe_extra_args
        )
        parents = self.pcmci.return_parents_dict(
            graph=results['graph'], 
            val_matrix=results['val_matrix'],
            include_lagzero_parents=True
        )
        return parents


class LocalizedResidualHSIC:
    """
    Custom Tigramite-compatible Conditional Independence Test.
    Partials out Z locally per regime, applies HSIC, and aggregates using the median.
    """
    def __init__(self, data: np.ndarray, u: Optional[np.ndarray] = None):
        self.data = data
        self.u = u
        self.measure = "Localized_Residual_HSIC"
        self.confidence = False  # Tigramite requires this attribute

    def set_dataframe(self, dataframe) -> None:
        """
        Tigramite calls this method on CI tests during PCMCI initialization.
        Keep a reference and synchronize the raw 2D data matrix used in run_test.
        """
        self.dataframe = dataframe

        values = getattr(dataframe, "values", None)
        if isinstance(values, np.ndarray):
            if values.ndim == 2:
                self.data = values
            elif values.ndim >= 3:
                # Tigramite can store multiple datasets as a stacked array.
                self.data = values[0]
            return

        if isinstance(values, list) and len(values) > 0 and isinstance(values[0], np.ndarray):
            self.data = values[0]

    def run_test(self, X: list, Y: list, Z: list = [], tau_max: int = 0, alpha_or_thres: float = 0.05, **kwargs):
        # Tigramite passes lists of tuples: [(var_idx, lag)]. Lags are <= 0.
        all_lags = [lag for var, lag in X + Y + Z]
        min_lag = min(all_lags) if all_lags else 0
        start_t = abs(min_lag)
        T = self.data.shape[0]

        if start_t >= T - 5:
            return 0.0, 1.0, False

        def get_lagged_data(var_lag_list):
            arrays = []
            for var, lag in var_lag_list:
                # lag is negative or zero
                arrays.append(self.data[start_t + lag : T + lag, var])
            if not arrays:
                return np.empty((T - start_t, 0))
            return np.column_stack(arrays)

        X_data = get_lagged_data(X)
        Y_data = get_lagged_data(Y)
        Z_data = get_lagged_data(Z) if Z else None

        # Determine regime masks
        if self.u is not None:
            u_sliced = self.u[start_t : T]
            if u_sliced.ndim == 2:  # One-hot encoded matrix
                num_regimes = u_sliced.shape[1]
                regime_masks = [u_sliced[:, r] == 1 for r in range(num_regimes)]
            else:  # 1D categorical array
                regime_masks = [u_sliced == val for val in np.unique(u_sliced)]
        else:
            # Treat the entire dataset as a single regime
            regime_masks = [np.ones(T - start_t, dtype=bool)]

        p_values = []
        stats = []

        for mask in regime_masks:
            # Require minimum sample size to avoid mathematically unstable regressions/HSIC
            if np.sum(mask) < 6:
                continue

            X_local = X_data[mask]
            Y_local = Y_data[mask]

            if Z_data is not None and Z_data.shape[1] > 0:
                Z_local = Z_data[mask]
                
                # Partial out Z locally
                reg_x = LinearRegression().fit(Z_local, X_local)
                res_x = X_local - reg_x.predict(Z_local)
                
                reg_y = LinearRegression().fit(Z_local, Y_local)
                res_y = Y_local - reg_y.predict(Z_local)
            else:
                res_x = X_local
                res_y = Y_local

            # Apply HSIC Test
            # Lazy import avoids circular import between micro and group discovery modules.
            from group_causation.group_causal_discovery.group_resit import HSIC_Test
            stat, pval = HSIC_Test.test(res_x, res_y)
            p_values.append(pval)
            stats.append(stat)

        # Aggregate results
        if not p_values:
            return 0.0, 1.0, False

        combined_p_value = float(np.median(p_values))
        mean_stat = float(np.mean(stats))
        dependent = combined_p_value <= alpha_or_thres

        return mean_stat, combined_p_value, dependent

    def get_model_selection_criterion(self, j, parents, tau_max):
        raise NotImplementedError("Auto-alpha selection is unsupported for Localized HSIC.")


def convert_to_tigramite_dataframe(data: Union[TimeSeriesData, np.ndarray]) -> tigramite.data_processing.DataFrame:
    '''
    Convert the data to tigramite dataframe format
    Note: It only works if there is only one data array in the data object
    '''
    if isinstance(data, TimeSeriesData):
        names = data.var_names
        data_arrays = data.data_arrays
        dataframe = tigramite.data_processing.DataFrame(data_arrays[0], var_names=names)
    
    elif isinstance(data, np.ndarray):
        dataframe = tigramite.data_processing.DataFrame(data, var_names=[str(i) for i in range(data.shape[1])])
    
    return dataframe