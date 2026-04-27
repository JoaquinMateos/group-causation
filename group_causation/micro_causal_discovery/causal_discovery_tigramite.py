from typing import Optional, Union, Any
import numpy as np
import logging
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

import tigramite
import tigramite.data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.gpdc import GPDC
from tigramite.pcmci import PCMCI

from causalai.data.time_series import TimeSeriesData
from group_causation.micro_causal_discovery.micro_causal_discovery_base import MicroCausalDiscovery


class PCMCIWrapper(MicroCausalDiscovery):
    def __init__(self, data: np.ndarray, cond_ind_test='parcorr',
                 min_lag=1, max_lag=3, pc_alpha: float = 0.5, 
                 non_stationarity_info: Optional[dict[str, Any]] = None, 
                 num_generated_regimes_if_no_shift_info: int=1,
                 **kwargs):
        super().__init__(data, **kwargs)
        
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.pc_alpha = pc_alpha
        self.extra_args = kwargs
        non_stationarity_info = non_stationarity_info or {}

        self.u = None
        if cond_ind_test.startswith('shift_based_local_') and non_stationarity_info.get('type') == 'regime_shifts':
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
                self.u = np.zeros((T_data, num_regimes))
                self.u[np.arange(T_data), u_aligned] = 1
                
        elif cond_ind_test.startswith('localized_'):
            num_regimes = num_generated_regimes_if_no_shift_info
            logging.info(f"Assuming {num_regimes} regimes for localized test without explicit regime shift info.")
            T_data = data.shape[0]
            chunk_size = int(np.ceil(T_data / num_regimes))
            
            u_aligned = np.repeat(np.arange(num_regimes), chunk_size)[:T_data]
            self.u = np.zeros((T_data, num_regimes))
            self.u[np.arange(T_data), u_aligned] = 1

        if cond_ind_test in ['shift_based_local_hsic', 'localized_hsic']:
            self.cond_ind_test = LocalizedResidualTest(self._data, self.u, test_type='hsic')
        elif cond_ind_test in ['shift_based_local_parcorr', 'localized_parcorr']:
            self.cond_ind_test = LocalizedResidualTest(self._data, self.u, test_type='parcorr')
        else:
            self.cond_ind_test = {
                'parcorr': ParCorr(),
                'gpdc': GPDC(),
                'cmiknn': CMIknn(significance='fixed_thres'),
            }[cond_ind_test]
        
        logging.info(f"Initialized PCMCIWrapper with cond_ind_test={cond_ind_test}, min_lag={min_lag}, max_lag={max_lag}, pc_alpha={pc_alpha}")
        
        dataframe = convert_to_tigramite_dataframe(self._data)
        self.pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=self.cond_ind_test,
            verbosity=0,
        )
    
    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        safe_extra_args = {k: v for k, v in self.extra_args.items() if k != 'num_regimes'}
        
        results = self.pcmci.run_pcmciplus(
            tau_min=self.min_lag, 
            tau_max=self.max_lag,
            pc_alpha=self.pc_alpha,
            **safe_extra_args
        )
        return self.pcmci.return_parents_dict(
            graph=results['graph'], 
            val_matrix=results['val_matrix'],
            include_lagzero_parents=True
        )


class LocalizedResidualTest:
    def __init__(self, data: np.ndarray, u: Optional[np.ndarray] = None, test_type: str = 'hsic'):
        self.data = data
        self.u = u
        self.test_type = test_type
        self.measure = f"Localized_Residual_{test_type.upper()}"
        self.confidence = False

    def set_dataframe(self, dataframe) -> None:
        self.dataframe = dataframe
        values = getattr(dataframe, "values", None)
        if isinstance(values, np.ndarray):
            if values.ndim == 2:
                self.data = values
            elif values.ndim >= 3:
                self.data = values[0]
            return

        if isinstance(values, list) and len(values) > 0 and isinstance(values[0], np.ndarray):
            self.data = values[0]

    def run_test(self, X: list, Y: list, Z: list = [], tau_max: int = 0, alpha_or_thres: float = 0.05, **kwargs):
        all_lags = [lag for var, lag in X + Y + Z]
        min_lag = min(all_lags) if all_lags else 0
        start_t = abs(min_lag)
        T = self.data.shape[0]

        if start_t >= T - 5:
            return 0.0, 1.0, False

        def get_lagged_data(var_lag_list):
            arrays = [self.data[start_t + lag : T + lag, var] for var, lag in var_lag_list]
            return np.column_stack(arrays) if arrays else np.empty((T - start_t, 0))

        X_data = get_lagged_data(X)
        Y_data = get_lagged_data(Y)
        Z_data = get_lagged_data(Z) if Z else None

        if self.u is not None:
            u_sliced = self.u[start_t : T]
            if u_sliced.ndim == 2:
                num_regimes = u_sliced.shape[1]
                regime_masks = [u_sliced[:, r] == 1 for r in range(num_regimes)]
            else:
                regime_masks = [u_sliced == val for val in np.unique(u_sliced)]
        else:
            regime_masks = [np.ones(T - start_t, dtype=bool)]

        p_values = []
        stats = []

        for mask in regime_masks:
            if np.sum(mask) < 20:
                continue

            X_local = X_data[mask]
            Y_local = Y_data[mask]

            if Z_data is not None and Z_data.shape[1] > 0:
                Z_local = Z_data[mask]
                
                reg_x = LinearRegression().fit(Z_local, X_local)
                res_x = X_local - reg_x.predict(Z_local)
                
                reg_y = LinearRegression().fit(Z_local, Y_local)
                res_y = Y_local - reg_y.predict(Z_local)
            else:
                res_x = X_local
                res_y = Y_local

            if self.test_type == 'hsic':
                from group_causation.group_causal_discovery.group_resit import HSIC_Test
                stat, pval = HSIC_Test.test(res_x, res_y)
            else:
                stat, pval = pearsonr(res_x.flatten(), res_y.flatten())

            p_values.append(pval)
            stats.append(stat)

        if not p_values:
            return 0.0, 1.0, False

        combined_p_value = float(np.median(p_values))
        mean_stat = float(np.mean(stats))
        dependent = combined_p_value <= alpha_or_thres

        return mean_stat, combined_p_value, dependent

    def get_model_selection_criterion(self, j, parents, tau_max):
        raise NotImplementedError(f"Auto-alpha selection is unsupported for {self.measure}.")


def convert_to_tigramite_dataframe(data: Union[TimeSeriesData, np.ndarray]) -> tigramite.data_processing.DataFrame:
    if isinstance(data, TimeSeriesData):
        return tigramite.data_processing.DataFrame(data.data_arrays[0], var_names=data.var_names)
    elif isinstance(data, np.ndarray):
        return tigramite.data_processing.DataFrame(data, var_names=[str(i) for i in range(data.shape[1])])