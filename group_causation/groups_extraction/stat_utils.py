from typing import Callable
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score


def get_pc1_explained_variance(data: np.ndarray) -> float:
    '''
    Get the explained variance of the first principal component of the data.
    '''
    if data.shape[1] == 0:
        return 1
    pca = PCA(n_components=1)
    pca.fit(data)
    
    return pca.explained_variance_ratio_[0]

def get_average_pc1_explained_variance(data: np.ndarray, groups: list[set[int]]) -> float:
    '''
    Get the average explained variance of the first principal component of the data
    for each group.
    '''
    explained_variances = [get_pc1_explained_variance(data[:, list(group)]) for group in groups]
    
    return float(np.mean(explained_variances))

def get_harmonic_explained_variance(data: np.ndarray, groups: list[set[int]]) -> float:
    '''
    Get the average explained variance of the first principal component of the data
    for each group.
    '''
    explained_variances = np.array([get_pc1_explained_variance(data[:, list(group)]) for group in groups])

    return float(1 / np.mean(1 / explained_variances)) if np.all(explained_variances > 1e-6) else 0.0

def get_explainability_score(data: np.ndarray, groups: list[set[int]]) -> float:
    '''
    Get a score that represents how well the data can be explained by the groups.
    '''
    cleaned_groups = [group for group in groups if len(group) > 0]
    
    explained_variance = get_harmonic_explained_variance(data, cleaned_groups)
    inverse_n_groups = 1 - len(cleaned_groups) / data.shape[1]
    
    geometric_mean = (explained_variance * inverse_n_groups) ** (1/2)
    
    return geometric_mean

def get_normalized_mutual_information(pred_groups: list[set[int]], gt_groups: list[set[int]]):
    '''
    Get the normalized mutual information between two sets of groups.
    '''
    n_nodes = len([*pred_groups])
    
    # Adapt group format to the labels one required by sklearn
    pred_labels = np.zeros(n_nodes)
    for i in range(n_nodes):
        for j, group in enumerate(pred_groups):
            if i in group:
                pred_labels[i] = j
                break
    
    gt_labels = np.zeros(n_nodes)
    for i in range(n_nodes):
        for j, group in enumerate(gt_groups):
            if i in group:
                gt_labels[i] = j
                break
    
    return normalized_mutual_info_score(gt_labels, pred_labels)

def get_scores_getter(data: np.ndarray, scores: list[str]) -> Callable:
    '''
    Generate a score getter function that receives a set of groups and returns a score to maximize.
    '''
    scores_getters = {
        'average_variance_explained': get_pc1_explained_variance,
        'harmonic_variance_explained': get_harmonic_explained_variance,
        'explainability_score': get_explainability_score,
    }
    
    return lambda groups: [scores_getters[score](data, groups) for score in scores]
