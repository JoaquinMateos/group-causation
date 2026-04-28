import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import Any, Union
import logging

from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.group_causal_discovery.micro_level import MicroLevelGroupCausalDiscovery
from group_causation.independence_tests import conditional_independence_tests
from group_causation.independence_tests.conditional_independence_base import ConditionalIndependence_base

class HybridGroupCausalDiscovery(GroupCausalDiscovery):
    '''
    Class that implements a group causal discovery algorithm which combines dimension reduction
    techniques with microlevel causal discovery.
    
    Includes Adag (Adaptive Aggregation) optimization to automatically tune the PCA 
    latent dimensions based on the c_ind (independence consistency) score.
    '''
    def __init__(self, data: np.ndarray,
                 groups: list[set[int]],
                 dimensionality_reduction_params: dict[str, Any],
                 link_assumptions: Union[dict[int, dict[tuple[int, int], str]], None] = None,
                 dimensionality_reduction: str = 'pca',
                 node_causal_discovery_alg: str = 'pcmci',
                 node_causal_discovery_params: Union[dict[str, Any], None] = None,
                 apply_adag_optimization: bool = True,
                 conditional_independence_test_for_adag: str = 'max_corr',
                 pc_alpha_for_adag: float = 0.05,
                 target_c_ind: float = 0.85,
                 verbose: int = 0,
                 **kwargs):
        
        super().__init__(data, groups, **kwargs)
        
        self._node_causal_discovery_alg = node_causal_discovery_alg
        self._node_causal_discovery_params = node_causal_discovery_params if node_causal_discovery_params is not None else {}
        self._link_assumptions = link_assumptions
        self._dimensionality_reduction = dimensionality_reduction
        self._dimensionality_reduction_params = dimensionality_reduction_params
        self._verbose = verbose
        
        # Adag specific parameters
        self.apply_adag_optimization = apply_adag_optimization
        self.target_c_ind = target_c_ind
        self.pc_alpha = pc_alpha_for_adag
        
        if conditional_independence_test_for_adag not in conditional_independence_tests:
            raise ValueError(f"Unsupported independence test: {conditional_independence_test_for_adag}")
        self.ci_test: ConditionalIndependence_base = conditional_independence_tests[conditional_independence_test_for_adag]

        # Device mapping for raw data CI testing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Normalize data once
        if self._dimensionality_reduction == 'pca':
            self._data = (self._data - self._data.mean(axis=0))
            if np.all((std := self._data.std(axis=0)) != 0): 
                self._data /= std
        else:
            raise ValueError(f'Dimensionality reduction technique {dimensionality_reduction} not supported.')

        # Pre-slice raw data into tensors for fast Adag c_ind evaluation
        self._raw_group_data = [
            torch.tensor(self._data[:, list(group)], dtype=torch.float32, device=self.device) 
            for group in self._groups
        ]
        
        # Placeholders for final state (populated during extract_parents)
        self.micro_groups = None
        self.micro_data = None

    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        '''
        Extract the parents of each group of variables. Branches depending on Adag hyperparameter.
        '''
        if self.apply_adag_optimization:
            if self._verbose > 0:
                print(f"Extracting parents: Adag optimization ENABLED (target c_ind={self.target_c_ind}).")
            return self._run_adag()
        else:
            if self._verbose > 0:
                print("Extracting parents: Adag optimization DISABLED. Using fixed PCA parameters.")
            
            # Legacy initialization logic
            self.micro_groups, self.micro_data = self._prepare_micro_groups_pca(**self._dimensionality_reduction_params)
            self._setup_micro_cd_algorithm(self.micro_groups, self.micro_data)
            
            return self.micro_level_causal_discovery.extract_parents()

    def _run_adag(self) -> dict[int, list[tuple[int, int]]]:
        """Iterative aggregation loop to find the optimal PCA dimensionality."""
        N = len(self._groups)
        m = [1] * N
        max_m = [len(group) for group in self._groups]
        
        c_ind_score = 0.0
        final_parents = {}
        
        while c_ind_score < self.target_c_ind:
            if self._verbose > 0:
                print(f"Adag Iteration | Current PCA components m: {m}")
                
            # 1. Reduce dimensionality using exactly m components per group
            micro_groups, micro_data = self._prepare_micro_groups_pca_adag(m)
            
            # 2. Setup and run Micro-Level Causal Discovery
            self._setup_micro_cd_algorithm(micro_groups, micro_data)
            group_parents = self.micro_level_causal_discovery.extract_parents()
            
            # 3. Compute consistency score against raw data
            c_ind_score = self._compute_c_ind(group_parents)
            
            if self._verbose > 0:
                print(f"Target c_ind: {self.target_c_ind} | Achieved c_ind: {c_ind_score:.3f}")
            
            # 4. Save state
            final_parents = group_parents
            self.micro_groups = micro_groups
            self.micro_data = micro_data
            
            # 5. Check termination
            if c_ind_score >= self.target_c_ind or m == max_m:
                if self._verbose > 0:
                    print("Adag termination condition met.")
                break
                
            # 6. Increment dimensions
            for i in range(N):
                if m[i] < max_m[i]:
                    m[i] += 1
                    
        return final_parents

    def _setup_micro_cd_algorithm(self, micro_groups, micro_data):
        """Helper to safely re-initialize the inner CD algorithm with new micro data."""
        micro_link_assumptions = _convert_link_assumptions(self._link_assumptions, micro_groups)
        params = self._node_causal_discovery_params.copy()
        params['link_assumptions'] = micro_link_assumptions
        
        self.micro_level_causal_discovery = MicroLevelGroupCausalDiscovery(
            micro_data, micro_groups, self._node_causal_discovery_alg, params
        )

    def _compute_c_ind(self, group_parents: dict[int, list[tuple[int, int]]]) -> float:
        """
        Evaluates independence consistency on the raw un-aggregated data.
        Infers conditional independencies from the absence of edges in the discovered group graph.
        """
        C_ind_count = 0
        I_ind_count = 0
        N = len(self._groups)
        
        for j in range(N):
            # Extract all groups identified as parents of j (collapsing over time lags for the cond set)
            parents_of_j = list(set([p[0] for p in group_parents.get(j, [])]))
            
            for i in range(N):
                if i == j: 
                    continue
                    
                # If i is not a parent of j, the algorithm determined they are independent 
                # (or d-separated) given j's Markov blanket. We verify this on raw data.
                if i not in parents_of_j:
                    pval = self._test_ci_raw(i, j, parents_of_j)
                    
                    if pval > self.pc_alpha:
                        C_ind_count += 1
                    else:
                        I_ind_count += 1
                        
        total = C_ind_count + I_ind_count
        return C_ind_count / total if total > 0 else 1.0

    def _test_ci_raw(self, i: int, j: int, cond_groups: list[int]) -> float:
        """Tests conditional independence between raw high-dimensional groups."""
        X = self._raw_group_data[i]
        Y = self._raw_group_data[j]
        
        if cond_groups:
            Z = torch.cat([self._raw_group_data[k] for k in cond_groups], dim=1)
            _, pval = self.ci_test.conditional_test(X, Y, Z)
        else:
            _, pval = self.ci_test.test(X, Y)
            
        return pval

    def _prepare_micro_groups_pca_adag(self, m_list: list[int]) -> tuple[list[set[int]], np.ndarray]:
        """Specific PCA execution for the Adag loop, forcing exactly m components per group."""
        micro_groups = []
        micro_data = []
        current_number_of_variables = 0
        
        for idx, group in enumerate(self._groups):
            group_data = self._data[:, list(group)]
            m = m_list[idx]
            
            pca = PCA(n_components=m)
            group_data_pca = pca.fit_transform(group_data)
            
            n_variables = group_data_pca.shape[1]
            micro_group = set(range(current_number_of_variables, current_number_of_variables + n_variables))
            
            micro_groups.append(micro_group)
            micro_data.append(group_data_pca)
            current_number_of_variables += n_variables
            
        micro_data = np.concatenate(micro_data, axis=1)
        return micro_groups, micro_data

    def _prepare_micro_groups_pca(self, explained_variance_threshold: float = 0.5,
                                  embedding_ratio: Union[float, None] = None,
                                  embedding_size: Union[int, None] = None,
                                  groups_division_method: str='group_embedding') -> tuple[list[set[int]], np.ndarray]:
        '''
        Execute the PCA dimensionality reduction algorithm to the groups of variables,
        in order to obtain a univariate time series for each group.
        '''
        if embedding_ratio is not None and embedding_size is not None:
            raise ValueError('Only one of embedding_ratio or embedding_size can be specified.')
        if embedding_ratio is not None:
            self._explained_variance_threshold = self._get_variance_threshold_from_embedding_ratio_pca(embedding_ratio)
        elif embedding_size is not None:
            self._explained_variance_threshold = self._get_variance_threshold_from_embedding_size_pca(embedding_size)
        else:
            self._explained_variance_threshold = explained_variance_threshold
            
        # Admit a low error when explained_variance_threshold is 0.0
        if self._explained_variance_threshold == 0.0:
            self._explained_variance_threshold = 0.05

        if self._explained_variance_threshold < 0 or self._explained_variance_threshold >= 1:
            raise ValueError(f'Explained variance threshold must be between 0 and 1. Obtained: {self._explained_variance_threshold}.\n'
                             'Note that if you specified embedding_ratio, the explained variance threshold will be calculated from it.')
        else:
            self._explained_variance_threshold = float(self._explained_variance_threshold)
        
        micro_groups = []
        micro_data = [] # List where each element is the ts data of a microgroup
        for group in self._groups:
            
            if groups_division_method == 'group_embedding':
                current_number_of_variables = sum(arr.shape[1] for arr in micro_data)
                
                micro_group, group_data_pca = self._get_group_embedding(group, current_number_of_variables)
                micro_groups.append(micro_group)
                micro_data.append(group_data_pca)
                
            elif groups_division_method == 'subgroups':                
                micro_group, group_data_pca = self._divide_subgroups(group)
                micro_groups.append( set(micro_group) )
                micro_data.append(group_data_pca)
            
            else:
                raise ValueError(f'Invalid groups division method: {groups_division_method}')
        
        micro_data = np.concatenate(micro_data, axis=1)
        
        if self._verbose > 0:
            print(f'Data dimensionality has been reduced to {micro_data.shape[1]} in order to perform microlevel causal discovery.')

        return micro_groups, micro_data

    def _get_group_embedding(self, group: list[int], current_number_of_variables: int) -> tuple[set[int], np.ndarray]:
        group_data = self._data[:, list(group)]
                
        # Extract the principal components of the group
        pca = PCA(n_components=self._explained_variance_threshold)
        group_data_pca = pca.fit_transform(group_data)

        # Create the microgroup
        n_variables = group_data_pca.shape[1]
        micro_group = set(range(current_number_of_variables,
                                current_number_of_variables + n_variables))
        
        return micro_group, group_data_pca
    
    def _divide_subgroups(self, current_subgroup: list[int], current_number_of_variables: int = 0) -> tuple[list[int], np.ndarray]:
        current_subgroup_data = self._data[:, list(current_subgroup)]
        pca = PCA(n_components=1)
        group_data_pca = pca.fit_transform(current_subgroup_data)
        if pca.explained_variance_ratio_[0] >= self._explained_variance_threshold or len(current_subgroup) == 1:
            used_subgroup = [current_number_of_variables]
            current_number_of_variables += 1
            return used_subgroup, group_data_pca
        else:
            ordered_nodes = np.argsort(pca.components_[0])
            half = len(current_subgroup) // 2
            first_half = ordered_nodes[:half]
            second_half = ordered_nodes[half:]
            first_subgroup, first_subgroup_data = self._divide_subgroups(
                [current_subgroup[i] for i in first_half],
                current_number_of_variables,
            )
            second_subgroup, second_subgroup_data = self._divide_subgroups(
                [current_subgroup[i] for i in second_half],
                current_number_of_variables + len(first_subgroup),
            )
            return first_subgroup + second_subgroup, np.concatenate([first_subgroup_data, second_subgroup_data], axis=1)
    
    def _get_variance_threshold_from_embedding_ratio_pca(self, embedding_ratio: Union[float, None] = None) -> float:
        if embedding_ratio is None:
            raise ValueError('embedding_ratio must be provided when using embedding_ratio mode.')

        variance_thresholds = []
        for group in self._groups:
            group_data = self._data[:, list(group)]
            pca = PCA(n_components=int( embedding_ratio * len(group) ))
            pca.fit(group_data)
            explained_variance = pca.explained_variance_ratio_.sum()
            variance_thresholds.append(explained_variance)
        explained_variance_threshold = float(np.mean(variance_thresholds))
        
        return explained_variance_threshold
    
    def _get_variance_threshold_from_embedding_size_pca(self, embedding_size: Union[int, None] = None) -> float:
        if embedding_size is None:
            raise ValueError('embedding_size must be provided when using embedding_size mode.')

        variance_thresholds = []
        for group in self._groups:
            if embedding_size >= len(group):
                variance_thresholds.append(1)
                continue
            group_data = self._data[:, list(group)]
            pca = PCA(n_components=int( embedding_size ))
            pca.fit(group_data)
            explained_variance = pca.explained_variance_ratio_.sum()
            variance_thresholds.append(explained_variance)
        explained_variance_threshold = float(np.mean(variance_thresholds))
        
        return explained_variance_threshold

def _convert_link_assumptions(link_assumptions: Union[dict[int, dict[tuple[int, int], str]], None], micro_groups: list[set[int]]) -> Union[dict[int, dict[tuple[int, int], str]], None]:
    if link_assumptions is None:
        return None
    
    micro_link_assumptions = {}
    for son_group_idx, son_group in enumerate(micro_groups):
        for son_node_idx in son_group:
            if son_node_idx not in micro_link_assumptions:
                micro_link_assumptions[son_node_idx] = {}
            for (parent_group_idx, lag), link_type in link_assumptions[son_group_idx].items():
                for parent_node_idx in micro_groups[parent_group_idx]:
                    micro_link_assumptions[son_node_idx][(parent_node_idx, lag)] = link_type
    
    return micro_link_assumptions