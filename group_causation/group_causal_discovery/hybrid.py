import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import Any, Union, List, Tuple, Set, Optional
import logging

from group_causation.dimensionality_reduction.adag_wrapper import AdagWrapper, TunablePCA
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
                 apply_adag_optimization: bool = False,
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
        logging.info(f"Adag optimization is {'ENABLED' if apply_adag_optimization else 'DISABLED'}.")
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
            
            # 1. Initialize Adag Components
            adag_wrapper = AdagWrapper(
                ci_test_class=self.ci_test,
                groups=self._groups,
                max_lag=self._node_causal_discovery_params.get('max_lag', 5),
                p_val_threshold=self.pc_alpha,
                num_regimes=self._node_causal_discovery_params.get('num_regimes', 1)
            )
            aggregator = TunablePCA()
            
            # We track the last found parents to avoid running CD one extra time at the end
            last_parents = {}
            
            # 2. Define the callback function for Adag
            def discovery_func(Z_m: List[torch.Tensor]) -> dict[int, list[tuple[int, int]]]:
                nonlocal last_parents
                
                # Format Z_m back to the micro_data / micro_groups layout
                micro_groups, micro_data = self._format_z_m_to_micro(Z_m)
                
                # Save to class state
                self.micro_groups = micro_groups
                self.micro_data = micro_data
                
                # Run the inner algorithm
                self._setup_micro_cd_algorithm(micro_groups, micro_data)
                last_parents = self.micro_level_causal_discovery.extract_parents()
                return last_parents

            # 3. Execute Adag Loop
            Z_m, c_ind_score, final_m = adag_wrapper.run(
                X_data=self._raw_group_data,
                discovery_func=discovery_func,
                aggregator=aggregator,
                target_alpha_q=self.target_c_ind
            )
            
            if self._verbose > 0:
                print(f"Adag optimization finished with components m={final_m} and c_ind={c_ind_score:.3f}")
                
            return last_parents

        else:
            if self._verbose > 0:
                print("Extracting parents: Adag optimization DISABLED. Using fixed PCA parameters.")
            
            # Legacy initialization logic
            self.micro_groups, self.micro_data = self._prepare_micro_groups_pca(**self._dimensionality_reduction_params)
            self._setup_micro_cd_algorithm(self.micro_groups, self.micro_data)
            
            return self.micro_level_causal_discovery.extract_parents()

    def _format_z_m_to_micro(self, Z_m: List[torch.Tensor]) -> Tuple[List[Set[int]], np.ndarray]:
        """Converts the list of aggregated tensors back into the micro_groups/micro_data format."""
        micro_groups = []
        micro_data = []
        current_number_of_variables = 0
        
        for z in Z_m:
            z_np = z.cpu().numpy()
            n_variables = z_np.shape[1]
            
            micro_group = set(range(current_number_of_variables, current_number_of_variables + n_variables))
            micro_groups.append(micro_group)
            micro_data.append(z_np)
            
            current_number_of_variables += n_variables
            
        micro_data_concat = np.concatenate(micro_data, axis=1)
        return micro_groups, micro_data_concat

    def _setup_micro_cd_algorithm(self, micro_groups, micro_data):
        """Helper to safely re-initialize the inner CD algorithm with new micro data."""
        micro_link_assumptions = _convert_link_assumptions(self._link_assumptions, micro_groups)
        params = self._node_causal_discovery_params.copy()
        params['link_assumptions'] = micro_link_assumptions
        
        self.micro_level_causal_discovery = MicroLevelGroupCausalDiscovery(
            micro_data, micro_groups, self._node_causal_discovery_alg, params
        )

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