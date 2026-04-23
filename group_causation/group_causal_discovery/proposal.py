import numpy as np
from typing import Any, Union
from sklearn.linear_model import LinearRegression
import logging

from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.group_causal_discovery.iVAE.wrappers import IVAE_wrapper
from group_causation.group_causal_discovery.micro_level import MicroLevelGroupCausalDiscovery





class IVAEProposalCausalDiscovery(GroupCausalDiscovery):
    '''
    Group causal discovery algorithm combining iVAE for identifiable dimension 
    reduction and deconfounding, followed by microlevel causal discovery (e.g., PCMCI).
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        u : np.array with the auxiliary variable for iVAE, shape (n_samples, n_aux_vars).
            This is required for the conditionally factorized prior[cite: 101].
        groups : list[set[int]] list with the sets that will compound each group of variables.
        global_latent_dim : int, the number of latents to extract for the global confounding representation.
        group_latent_dims : Union[int, list[int]], dimensionality for each group's latent space.
        ivae_params : dict with parameters for the iVAE architecture.
        node_causal_discovery_alg : str indicating the algorithm used to discover causal relationships.
        node_causal_discovery_params : dict with parameters for the node causal discovery algorithm.
        link_assumptions : Dictionary specifying assumptions about links.
    '''
    def __init__(self, 
                 data: np.ndarray,
                 u: np.ndarray,
                 groups: list[set[int]],
                 global_latent_dim: int = 5,
                 group_latent_dims: Union[int, list[int]] = 2,
                 ivae_params: Union[dict[str, Any], None] = None,
                 link_assumptions: Union[dict[int, dict[tuple[int, int], str]], None] = None,
                 node_causal_discovery_alg: str = 'pcmci',
                 node_causal_discovery_params: Union[dict[str, Any], None] = None,
                 verbose: int = 0,
                 **kwargs):
        
        super().__init__(data, groups, **kwargs)
        
        self.u = u
        self._global_latent_dim = global_latent_dim
        
        if isinstance(group_latent_dims, int):
            self._group_latent_dims = [group_latent_dims] * len(self._groups)
        else:
            self._group_latent_dims = group_latent_dims
            
        self._ivae_params = ivae_params if ivae_params is not None else {}
        self._node_causal_discovery_alg = node_causal_discovery_alg
        self._node_causal_discovery_params = node_causal_discovery_params if node_causal_discovery_params is not None else {}
        self._verbose = verbose
        
        # Step 1 & 2: Extract representations and deconfound
        self.micro_groups, self.micro_data = self._prepare_micro_groups_ivae()
        
        # Step 3: Setup Micro-Level Causal Discovery (PCMCI)
        micro_link_assumptions = self._convert_link_assumptions(link_assumptions, self.micro_groups)
        self._node_causal_discovery_params['link_assumptions'] = micro_link_assumptions
        
        self.micro_level_causal_discovery = MicroLevelGroupCausalDiscovery(
            self.micro_data, 
            self.micro_groups,
            self._node_causal_discovery_alg, 
            self._node_causal_discovery_params
        )

    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        '''
        Extract the parents of each group of variables using the dimension reduction algorithm.
        '''
        if self._verbose > 0:
            logging.info("Extracting parents using micro-level causal discovery on iVAE latents.")
        group_parents = self.micro_level_causal_discovery.extract_parents()
        return group_parents

    def _prepare_micro_groups_ivae(self) -> tuple[list[set[int]], np.ndarray]:
        '''
        Extracts a global representation, reduces dimensionality of each group, 
        and deconfounds the local representations.
        '''
        # 1. Extract global latent representation of all groups
        if self._verbose > 0:
            logging.info(f"Training global iVAE")
        
        global_latents, model, params, history = IVAE_wrapper(self._data, self.u, **self._ivae_params)
        logging.info(f"Number of global latent variables: {global_latents.shape[1]}")
        micro_groups = []
        micro_data_list = []
        current_number_of_variables = 0
        
        # 2. Extract group specific representations and deconfound
        for idx, group in enumerate(self._groups):
            group_data = self._data[:, list(group)]
            group_dim = self._group_latent_dims[idx]
            
            # Reduce dimensionality of the specific group using iVAE
            group_latents, model, params, history = IVAE_wrapper(group_data, self.u, **self._ivae_params)
            logging.info(f"Number of latent variables for group {idx}: {group_latents.shape[1]}")

            # Deconfound: remove linear dependence on the global latents
            # (Assuming linear relationships in the latent space for simplification)
            deconfounder = LinearRegression()
            deconfounder.fit(global_latents, group_latents)
            predicted_confounding = deconfounder.predict(global_latents)
            
            # The residuals act as our deconfounded latent variables
            deconfounded_group_latents = group_latents - predicted_confounding
            
            # Bookkeeping for micro_groups
            n_variables = deconfounded_group_latents.shape[1]
            micro_group = set(range(current_number_of_variables, current_number_of_variables + n_variables))
            
            micro_groups.append(micro_group)
            micro_data_list.append(deconfounded_group_latents)
            
            current_number_of_variables += n_variables
            
        micro_data = np.concatenate(micro_data_list, axis=1)
        
        if self._verbose > 0:
            logging.info(f'Data dimensionality reduced to {micro_data.shape[1]} (deconfounded iVAE latents).')

        return micro_groups, micro_data

    def _convert_link_assumptions(self, link_assumptions: Union[dict[int, dict[tuple[int, int], str]], None], micro_groups: list[set[int]]) -> Union[dict[int, dict[tuple[int, int], str]], None]:
        '''
        Convert the link assumptions from the original groups to the microgroups.
        '''
        if link_assumptions is None:
            return None
        
        micro_link_assumptions = {}
        for son_group_idx, son_group in enumerate(micro_groups):
            for son_node_idx in son_group:
                if son_node_idx not in micro_link_assumptions:
                    micro_link_assumptions[son_node_idx] = {}
                
                # Check if the son group has any link assumptions assigned to it
                if son_group_idx in link_assumptions:
                    for (parent_group_idx, lag), link_type in link_assumptions[son_group_idx].items():
                        for parent_node_idx in micro_groups[parent_group_idx]:
                            micro_link_assumptions[son_node_idx][(parent_node_idx, lag)] = link_type
        
        return micro_link_assumptions
