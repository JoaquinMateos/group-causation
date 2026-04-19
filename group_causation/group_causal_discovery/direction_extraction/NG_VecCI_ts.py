import copy
import numpy as np
from typing import Any
from group_causation.utils import get_precision, get_recall
from group_causal_discovery.direction_extraction.direction_extraction_base import DirectionExtractorBase, EdgeDirection


import tigramite.data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.pcmci import PCMCI
from sklearn.linear_model import LinearRegression

class NG_VecCI(DirectionExtractorBase):
    '''
    Class that implements the NG-Vector Causal Inference algorithm for direction extraction on groups of variables.
    This is an extension of the 2G-Vector Causal Inference algorithm for groups of variables,
    [Wahl, J., Ninad, U., & Runge, J. (2023, June). Vector causal inference between two groups of variables.]
    '''
    def identify_causal_direction(self, X: np.ndarray, Y: np.ndarray, lag_X=0, alpha=0.01,
                                    CI_test_method='ParCorr', ambiguity = None,
                                    test = 'full', max_sep_set = None, linear = True,
                                    fit_intercept = False, random_state = None) -> tuple[EdgeDirection, dict[str, Any]]:
        '''
        Function that executes either 2G-VecCI.PC or 2G-VecCi.Full and outputs the causal direction as well as the inference criterion
        and details on edge counts and densities
        
        Args:
            X : tigramite dataframe of form (T,N) where T is sample and N is group size
            Y : tigramite dataframe of form (T,N') where T is sample and N' is group size
            alpha : significance level of CI tests
            CI_test_method : conditional independence test method, default ParCorr
            ambiguity : ambiguity level as specified in cited paper, if None chosen to be alpha
            test : string, either 'full' for 2G-VecCI.Full or 'PC' for 2G-VecCI.PC
            max_sep_set : maximal size of d-separation set in case 2G-VecCI.PC is used
            linear : boolean, tests for linear or non-linear interactions
            fit_intercept : boolean, indicates whether an intercept should be fitted in regression methods, default is False
        
        Returns:
            comparison : string indicating the inferred causal direction
            cause_criterion : Crit, as specified in the submitted paper
            test_results : details on edge densities and counts
        '''
        if ambiguity == None:
            ambiguity = alpha

        if test == 'full':
            test_results = _full_conditioning_ind_test(X, Y, lag_X=lag_X, max_lag=self.max_lag, alpha=alpha,
                                    CI_test_method=CI_test_method, linear = linear,
                                    fit_intercept=fit_intercept, random_state=random_state)
        # elif test == 'PC':
        #     test_results = conditioning_ind_test_with_PC(X, Y, alpha, CI_test_method=CI_test_method, max_sep_set=max_sep_set, linear_interactions=linear)
        
        X_criterion = test_results['edge density of Res X'] - test_results['edge density of X']
        Y_criterion = test_results['edge density of Res Y'] - test_results['edge density of Y']
        criterion = X_criterion - Y_criterion
        # APPLY THEOREM ABOUT CRITERION TO INFER DIRECTION
        if abs(criterion) < ambiguity:
            direction = EdgeDirection.NONE
        elif criterion > ambiguity:
            direction = EdgeDirection.LEFT2RIGHT
        elif criterion < -ambiguity:
            direction = EdgeDirection.RIGHT2LEFT
            
        return direction, test_results


def _extract_pvalue(test_output: Any) -> float:
    if isinstance(test_output, tuple) and len(test_output) >= 2 and test_output[1] is not None:
        return float(test_output[1])
    return 1.0
        



def _full_conditioning_ind_test(X: np.ndarray, Y: np.ndarray, lag_X=0, max_lag=3, alpha=0.01,
                                CI_test_method='ParCorr', linear = True,
                                fit_intercept = False, random_state = None):
    '''
    Implementation of 2G-VecCI.Full as desribed in the submitted article [https://github.com/JonasChoice/2GVecCI], but extended to time series.
    Runs sparsity based independent test of regions X and Y with the prescribed CI_test_method.
    
    Args:
        X: np.ndarray containing posible origin variables, shape (T, N)
        Y: np.ndarray containing possible target variables, shape (T, N')
        lag_X: integer, the lag of the regression (we regress Y on X with lag lag_X)
        max_lag: integer, maximum lag to consider in the test
        alpha: floating number, significance level for conditional independence testing
        CI_test_method: The conditional independence test. Options: 'ParCorr', 'GPDC', 'CMIknn', see the documentation
            of Tigramite for details on the implementations of these tests
        linear: string, Options: 'yes' or 'no'. If yes, conditioning on regions is handled by an OLS regression step and further
            tests are run on the residuals . If no, X is added to all conditioning sets when testing for Y|X densities and vice versa.
    
    Returns:
        dictionary describing number of detected edges on regions and their residuals as well as sparsity,
        measured as number of detected edges/number of edges on fully connected graph
    '''
    if random_state == None:
        random_state = np.random

    if CI_test_method == 'ParCorr':
        CI_test = ParCorr()
    if CI_test_method == 'GPDC':
        CI_test = GPDCtorch()
    if CI_test_method == 'CMIknn':
        CI_test = CMIknn()
    dict = {}
    # LINEAR CASE, where we can use residuals to test for conditional independence
    if linear == True:
        # OBTAIN EDGE DENSITY OF Y AND RES Y|X
        Regression = _regression(X, lag_X, Y)
        residualsY = Regression['X_to_Y']
        edgecounterY = 0
        edgecounterResY = 0
        #Set conditional independece tests data
        CI_test_Y = copy.deepcopy(CI_test)
        CI_test_Y.set_dataframe(pp.DataFrame(Y))
        CI_test_Y_residuals = copy.deepcopy(CI_test)
        CI_test_Y_residuals.set_dataframe(pp.DataFrame(residualsY))
        
        # Maximum number of edges     = contemporary edges        + lagged edges 
        get_max_edges = lambda N_vars : N_vars * (N_vars - 1) / 2 + N_vars**2 * max_lag
        max_edgenumberX = get_max_edges(X.shape[1])
        max_edgenumberY = get_max_edges(Y.shape[1])
        
        # Create lists of all possible variables and respective lags
        X_vars = [(var, -lag) for var in range(X.shape[1]) for lag in range(max_lag+1)]
        Y_vars = [(var, -lag) for var in range(Y.shape[1]) for lag in range(max_lag+1)]
        
        # iterate over all possible pairs of variables of Y, i.e., all possible edges
        for var1 in range(Y.shape[1]):
            for var2 in range(var1 + 1, Y.shape[1]):
                for lag in range(0, max_lag+1):
                    # Test the link (var1, -lag) -> (var2, 0)
                    cond_set = Y_vars.copy()
                    cond_set.remove((var1, -lag))
                    cond_set.remove((var2, 0))
                    
                    pvalY = _extract_pvalue(CI_test_Y.run_test([(var1, -lag)], [(var2, -lag)], cond_set))
                    if pvalY < alpha: # Is current edge significant over G_Y?
                        edgecounterY += 1
                    
                    pvalResY = _extract_pvalue(CI_test_Y_residuals.run_test([(var1, -lag)], [(var2, -lag)], cond_set))
                    if pvalResY < alpha: # Is current edge significant over G_{Y|X}?
                        edgecounterResY += 1
                
        dict['number of nodes Y'] = Y.shape[1]
        dict['max number of edges Y'] = max_edgenumberY
        dict['number of edges Y'] = edgecounterY
        dict['number of edges Res Y'] = edgecounterResY
        dict['edge density of Y'] = edgecounterY / max_edgenumberY
        dict['edge density of Res Y'] = edgecounterResY / max_edgenumberY
        
        # OBTAIN EDGE DENSITY OF X AND RES X|Y
        residualsX = Regression['Y_to_X']
        # Set conditional independece tests data
        CI_test_X = copy.deepcopy(CI_test)
        CI_test_X.set_dataframe(pp.DataFrame(X))
        CI_test_X_residuals = copy.deepcopy(CI_test)
        CI_test_X_residuals.set_dataframe(pp.DataFrame(residualsX))
        
        edgecounterX = 0
        edgecounterResX = 0
        # iterate over all possible pairs of variables, i.e., all possible edges
        for var1 in range(X.shape[1]):
            for var2 in range(var1 + 1, X.shape[1]):
                for lag in range(0, max_lag+1):
                    # Test the link (var1, -lag) -> (var2, 0)
                    cond_set = X_vars.copy()
                    cond_set.remove((var1, -lag))
                    cond_set.remove((var2, 0))
                    
                    pvalX = _extract_pvalue(CI_test_X.run_test([(var1, -lag)], [(var2, -lag)], cond_set))
                    if pvalX < alpha: # Is current edge significant over G_X?
                        edgecounterX += 1
                    
                    pvalResX = _extract_pvalue(CI_test_X_residuals.run_test([(var1, -lag)], [(var2, -lag)], cond_set))
                    if pvalResX < alpha: # Is current edge significant over G_{X|Y}?
                        edgecounterResX += 1
            
        dict['number of nodes X'] = X.shape[1]
        dict['max number of edges X'] = max_edgenumberX
        dict['number of edges X'] = edgecounterX
        dict['number of edges Res X'] = edgecounterResX
        dict['edge density of X'] = edgecounterX / max_edgenumberX
        dict['edge density of Res X'] = edgecounterResX / max_edgenumberX
        
        return dict
    
    else: # NON-LINEAR CASE, where we cannot use residuals
        # TODO: Adapt with case to time series
        edgecounterY = 0
        edgecounterResY = 0
        max_edgenumberY = Y.shape[1] * (Y.shape[1] - 1) / 2
        max_edgenumberX = X.shape[1] * (X.shape[1] - 1) / 2
        for var1 in range(Y.shape[1]):
            for var2 in range(var1 + 1, Y.shape[1]):
                removedY = np.delete(Y, (var1, var2), 1)
                pvalY = _extract_pvalue(CI_test.run_test_raw(Y[:, var1:var1 + 1], Y[:, var2:var2 + 1], z=removedY))
                if pvalY < alpha:
                    edgecounterY += 1
                cond_setY = np.concatenate((removedY,X),axis=1)
                pvalResY = _extract_pvalue(CI_test.run_test_raw(Y[:, var1:var1 + 1], Y[:, var2:var2 + 1],
                                                            z=cond_setY))
                if pvalResY < alpha:
                    edgecounterResY += 1
        dict['number of nodes Y'] = Y.shape[1]
        dict['max number of edges Y'] = max_edgenumberY
        dict['number of edges Y'] = edgecounterY
        dict['number of edges Res Y'] = edgecounterResY
        dict['edge density of Y'] = edgecounterY / max_edgenumberY
        dict['edge density of Res Y'] = edgecounterResY / max_edgenumberY
        edgecounterX = 0
        edgecounterResX = 0
        for var1 in range(X.shape[1]):
            for var2 in range(var1 + 1, X.shape[1]):
                removedX = np.delete(X, (var1, var2), 1)
                pvalX = _extract_pvalue(CI_test.run_test_raw(X[:, var1:var1 + 1], X[:, var2:var2 + 1], z=removedX))
                if pvalX < alpha:
                    edgecounterX += 1
                cond_setX = np.concatenate((removedX, Y), axis=1)
                pvalResX = _extract_pvalue(CI_test.run_test_raw(X[:, var1:var1 + 1], X[:, var2:var2 + 1],
                                                            z=cond_setX))
                if pvalResX < alpha:
                    edgecounterResX += 1
        dict['number of nodes X'] = X.shape[1]
        dict['max number of edges X'] = max_edgenumberX
        dict['number of edges X'] = edgecounterX
        dict['number of edges Res X'] = edgecounterResX
        dict['edge density of X'] = edgecounterX / max_edgenumberX
        dict['edge density of Res X'] = edgecounterResX / max_edgenumberX
        dict['tested regions'] = type
        
        return dict
    
    
def _regression(X: np.ndarray, lag_X: int, Y: np.ndarray):
    '''
    Regression function that regresses the random vector X linearly on Y and Y on X and returns both residuals
    
    Args:
        X: numpy array with shape (T, N) where T is the number of samples and N the number of variables
        lag_X: integer, the lag of the regression (we regress Y on X with lag lag_X)
        Y: numpy array with shape (T, N') where T is the number of samples and N' the number of variables
    
    Returns:
        dictionary containing the residuals of the regression of Y on X (key 'X_to_Y') and X on Y (key 'Y_to_X').
    '''
    dict = {}
    
    # Create lagged data
    X = X[lag_X:, :]
    Y = Y[:X.shape[0], :]
    
    # Perform linear regression
    reg_X_to_Y = LinearRegression(fit_intercept=False)
    reg_Y_to_X = LinearRegression(fit_intercept=False)
    reg_X_to_Y.fit(X, Y)
    reg_Y_to_X.fit(Y, X)
    
    # Obtain residuals
    dict['X_to_Y'] = Y - reg_X_to_Y.predict(X)
    dict['Y_to_X'] = X - reg_Y_to_X.predict(Y)
    
    return dict


if __name__ == '__main__':
    # TEST WITH DIFFERENT KINDS OF DATA
    np.random.seed(0)
    from group_causation.data_management.create_toy_datasets import CausalDataset
    def test_edge_extraction_capacity():
        '''
        Test the capacity of the edge extraction algorithm to extract edges from a time series
        '''
        max_lag = 3
        dataset = CausalDataset()
        ts_data, groups_parents_dict, groups, node_parents_dict = dataset.generate_group_toy_data(1, T=2000, N_vars=20, N_groups=4, noise_sigmas=[0.2],
                                                                                                  outer_group_crosslinks_density=0.8, min_lag=0, max_lag=max_lag,
                                                                                                  contemp_fraction=0.5, dependency_coeffs=[-0.3, 0.3],
                                                                                                  auto_coeffs=[0.], noise_dists=['gaussian'],
                                                                                                  dependency_funcs=['linear'])
        
        ng_vecci = NG_VecCI(ts_data, groups=[set(group) for group in groups], ambiguity=0.1, max_lag=max_lag)
        
        print('Start testing')
        predicted_groups_parents = {i: [] for i in range(len(ng_vecci._groups))}
        for i in range(len(ng_vecci._groups)):
            for j in range(i, len(ng_vecci._groups)):
                for lag in range(max_lag):
                    if i == j and lag == 0: # Do not test X->X
                        continue
                    direction = ng_vecci.extract_direction(i, j, lag)
                    if direction == EdgeDirection.LEFT2RIGHT:
                        predicted_groups_parents[j].append((i, -lag))
                    elif direction == EdgeDirection.RIGHT2LEFT:
                        predicted_groups_parents[i].append((j, -lag))
                    elif direction == EdgeDirection.BIDIRECTED:
                        predicted_groups_parents[j].append((i, -lag))
                        predicted_groups_parents[i].append((j, -lag))
        
        print(f'{groups_parents_dict=}')
        print(f'{predicted_groups_parents=}')
        prec = get_precision(groups_parents_dict, predicted_groups_parents)
        rec = get_recall(groups_parents_dict, predicted_groups_parents)
        print(f'Precision: {prec}, Recall: {rec}')
        
    
    test_edge_extraction_capacity()