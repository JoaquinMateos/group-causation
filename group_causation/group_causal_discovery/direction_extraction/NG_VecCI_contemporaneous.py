import numpy as np
from typing import Any
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
    def identify_causal_direction(self, X: np.ndarray, Y: np.ndarray, alpha=0.01,
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
            ambiguity : ambiguity level as specified in submitted paper, if None chosen to be alpha
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
            test_results = _full_conditioning_ind_test(
                X,
                Y,
                max_lag=self.max_lag,
                alpha=alpha,
                CI_test_method=CI_test_method,
                linear=linear,
                fit_intercept=fit_intercept,
                random_state=random_state,
            )
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
        



def _full_conditioning_ind_test(X: np.ndarray, Y: np.ndarray, max_lag=3, alpha=0.01,
                                CI_test_method='ParCorr', linear = True,
                                fit_intercept = False, random_state = None):
    '''
    Implementation of 2G-VecCI.Full as desribed in the submitted article [https://github.com/JonasChoice/2GVecCI].
    Runs sparsity based independent test of regions X and Y with the prescribed CI_test_method.
    
    Args:
        X: np.ndarray containing posible origin variables, shape (T, N)
        Y: np.ndarray containing possible target variables, shape (T, N')
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
        CI_test =ParCorr()
    if CI_test_method == 'GPDC':
        CI_test =GPDCtorch()
    if CI_test_method == 'CMIknn':
        CI_test =CMIknn()
    dict = {}
    # LINEAR CASE, where we can use residuals to test for conditional independence
    if linear == True:
        # OBTAIN EDGE DENSITY OF Y AND RES Y|X
        Regression = _regression(X, Y, fit_intercept=fit_intercept)
        residualsY = Regression['X_to_Y']
        edgecounterY = 0
        edgecounterResY = 0
        max_edgenumberY = Y.shape[1] * (Y.shape[1] - 1) / 2
        max_edgenumberX = X.shape[1] * (X.shape[1] - 1) / 2
        # iterate over all possible pairs of variables, i.e., all possible edges
        for var1 in range(Y.shape[1]):
            for var2 in range(var1 + 1, Y.shape[1]):
                removedY = np.delete(Y, (var1, var2), 1)
                pvalY = _extract_pvalue(CI_test.run_test_raw(Y[:, var1:var1 + 1], Y[:, var2:var2 + 1], z=removedY))
                
                if pvalY < alpha: # Is current edge significant over G_Y?
                    edgecounterY += 1
                
                removedResY = np.delete(residualsY, (var1, var2), 1)
                pvalResY = _extract_pvalue(CI_test.run_test_raw(residualsY[:, var1:var1 + 1], residualsY[:, var2:var2 + 1],
                                                            z=removedResY))
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
        edgecounterX = 0
        edgecounterResX = 0
        # iterate over all possible pairs of variables, i.e., all possible edges
        for var1 in range(X.shape[1]):
            for var2 in range(var1 + 1, X.shape[1]):
                removedX = np.delete(X, (var1, var2), 1)
                pvalX = _extract_pvalue(CI_test.run_test_raw(X[:, var1:var1 + 1], X[:, var2:var2 + 1], z=removedX))
                
                if pvalX < alpha: # Is current edge significant over G_X?
                    edgecounterX += 1
                
                removedResX = np.delete(residualsX, (var1, var2), 1)
                pvalResX = _extract_pvalue(CI_test.run_test_raw(residualsX[:, var1:var1 + 1], residualsX[:, var2:var2 + 1],
                                                            z=removedResX))
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
    
    
def _regression(X: np.ndarray, Y: np.ndarray, fit_intercept: bool=False):
    '''
    Regression function that regresses the random vector X linearly on Y and Y on X and returns both residuals
    
    Args:
        X: numpy array with shape (T, N) where T is the number of samples and N the number of variables
        Y: numpy array with shape (T, N') where T is the number of samples and N' the number of variables
        fit_intercept: boolean, whether to fit an intercept in the regression
    
    Returns:
        dictionary containing the residuals of the regression of Y on X (key 'X_to_Y') and X on Y (key 'Y_to_X').
    '''
    dict = {}
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
    
    def test_2G_VecCI():
        # # If data is random
        # X_vec = np.random.randn(100, 5)
        # Y_vec = np.random.randn(100, 5)
        
        # # If X -> Y
        # X_vec = np.random.randn(100, 5)
        # Y1 = X_vec[:, 0] + 0.5 * X_vec[:, 1] + np.random.normal(0, 0.1, 100)
        # Y2 = X_vec[:, 2] + 0.5 * X_vec[:, 3] + np.random.normal(0, 0.1, 100)
        # Y3 = X_vec[:, 4] + np.random.normal(0, 0.1, 100)
        # Y_vec = np.column_stack((Y1, Y2, Y3))
        
        # If X <- Y
        Y_vec = np.random.randn(100, 5)
        X1 = Y_vec[:, 0] + 0.5 * Y_vec[:, 1] + np.random.normal(0, 0.1, 100)
        X2 = Y_vec[:, 2] + 0.5 * Y_vec[:, 3] + np.random.normal(0, 0.1, 100)
        X3 = Y_vec[:, 4] + np.random.normal(0, 0.1, 100)
        X_vec = np.column_stack((X1, X2, X3))
        
        
        X_group = list(range(X_vec.shape[1]))
        Y_group = list(range(X_vec.shape[1], X_vec.shape[1]+Y_vec.shape[1]))
        
        data = np.concatenate((X_vec, Y_vec), axis=1)
        
        ng_vecci = NG_VecCI(data, groups=[set(X_group), set(Y_group)])
        
        direction, test_results = ng_vecci.identify_causal_direction(X_vec, Y_vec)
        
        print(direction)
    
    def test_NG_VecCI():
        '''
        I'm gonna test X -> Y <- Z, X -> Z, and see what directions we can infer
        '''
        X1 =      np.random.normal(0, 1, 1000).reshape(-1, 1)
        X2 = X1 + np.random.normal(0, 1, 1000).reshape(-1, 1)
        X3 = X2 + np.random.normal(0, 1, 1000).reshape(-1, 1)
        X = np.column_stack((X1, X2, X3))
        
        Z1 = X2 +      np.random.normal(0, 1, 1000).reshape(-1, 1)
        Z2 = X1 + Z1 + np.random.normal(0, 1, 1000).reshape(-1, 1)
        Z3 = Z2 +      np.random.normal(0, 1, 1000).reshape(-1, 1)
        Z = np.column_stack((Z1, Z2, Z3))
        
        Y1 = X1 +           np.random.normal(0, 1, 1000).reshape(-1, 1)
        Y2 = Z2 + X3 + np.random.normal(0, 1, 1000).reshape(-1, 1)
        Y3 = Y2 + X1 + Z3 + np.random.normal(0, 1, 1000).reshape(-1, 1)
        Y = np.column_stack((Y1, Y2, Y3))
        
        
        X_group = list(range(X.shape[1]))
        Y_group = list(range(X.shape[1], X.shape[1]+Y.shape[1]))
        Z_group = list(range(X.shape[1]+Y.shape[1], X.shape[1]+Y.shape[1]+Z.shape[1]))
        
        data = np.concatenate((X, Y, Z), axis=1)
        
        ng_vecci = NG_VecCI(data, groups=[set(X_group), set(Y_group), set(Z_group)])
        
        groups_dict = {0: 'X', 1: 'Y', 2: 'Z'}
        print('Start testing')
        for i in range(len(ng_vecci._groups)):
            for j in range(i+1, len(ng_vecci._groups)):
                direction = ng_vecci.extract_direction(i, j)
                print(f'For {groups_dict[i]}, {groups_dict[j]}, we get {direction}')
    
    # Execute the tests
    # test_2G_VecCI()
    test_NG_VecCI()
