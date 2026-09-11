import numpy as np
import itertools
import logging
import torch
import math
from scipy.stats import chi2
from typing import Any

from group_causation.group_causal_discovery.group_causal_discovery_base import GroupCausalDiscovery
from group_causation.independence_tests import conditional_independence_tests
from group_causation.independence_tests.conditional_independence_base import ConditionalIndependence_base

class GroupPCMCICausalDiscovery(GroupCausalDiscovery):
    '''
    Group-level causal discovery algorithm using the PCMCI framework.
    Includes flexible localized independence testing for non-stationary data, 
    allowing explicit regime shifts or uniform time-index chunking.
    '''
    def __init__(self,
                 data: np.ndarray,
                 groups: list[set[int]],
                 tau_max: int,
                 pc_alpha: float = 0.05,
                 max_conds_dim: int = 2,
                 u: np.ndarray | str | None = 'time_index',
                 conditional_independence_test: str = 'max_corr',
                 num_chunks_of_time_index: int | None = None,
                 pcmci_params: dict[str, Any] | None = None,
                 non_stationarity_info: dict[str, Any] | None = None,
                 enforce_causal_input_completeness: bool = False,
                 verbose: int = 0,
                 **kwargs):
        
        super().__init__(data, groups, **kwargs)
        
        if conditional_independence_test not in conditional_independence_tests:
            raise ValueError(f"Unsupported independence test: {conditional_independence_test}")
        self.conditional_independence_test: ConditionalIndependence_base = conditional_independence_tests[conditional_independence_test]
        
        self._pcmci_params = pcmci_params if pcmci_params is not None else {}
        self.non_stationarity_info = non_stationarity_info or {}
        self._verbose = verbose
        self._enforce_cic = enforce_causal_input_completeness
        
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.max_conds_dim = max_conds_dim
        
        self.device = self._get_device()
        self.u = None
        
        T_data = data.shape[0]

        # 1. Regime Construction / Time-Index Chunking
        if isinstance(u, str):
            if u == 'non_stationarity_shift':
                if self.non_stationarity_info.get('type') != 'regime_shifts':
                    raise ValueError("non_stationarity_info must have type 'regime_shifts' when u='non_stationarity_shift'")
                
                affected_vars = self.non_stationarity_info.get('affected_vars', [])
                if not affected_vars:
                    if self._verbose > 0:
                        logging.info("Notice: No variables affected by non-stationarity. Falling back to u='time_index'.")
                    u = 'time_index'
                else:
                    first_var = affected_vars[0]
                    shifts = self.non_stationarity_info['shift_details'][first_var]
                    
                    total_T = shifts[-1]['end']
                    u_full = np.zeros(total_T, dtype=int)
                    
                    for shift in shifts:
                        # Ensure 0-indexed mapping for regimes
                        regime_idx = shift['regime'] if shift['regime'] == 0 else shift['regime'] - 1
                        u_full[shift['start']:shift['end']] = regime_idx
                        
                    u_aligned = u_full[-T_data:]
                    num_regimes = self.non_stationarity_info.get('num_shifts', len(shifts)) + 1
                    
                    u_np = np.zeros((T_data, num_regimes))
                    u_np[np.arange(T_data), u_aligned] = 1
                    self.u = torch.tensor(u_np, dtype=torch.bool, device=self.device)

            if u == 'time_index':
                if num_chunks_of_time_index is None:
                    raise ValueError("num_chunks_of_time_index must be specified when u='time_index'")
                
                # Split uniformly across the time dimension
                chunk_indices = np.repeat(np.arange(num_chunks_of_time_index), 
                                          int(np.ceil(T_data / num_chunks_of_time_index)))[:T_data]
                u_np = np.zeros((T_data, num_chunks_of_time_index))
                u_np[np.arange(T_data), chunk_indices] = 1
                self.u = torch.tensor(u_np, dtype=torch.bool, device=self.device)
                
        elif isinstance(u, np.ndarray):
            self.u = torch.tensor(u, dtype=torch.bool, device=self.device)

        # 2. Pre-slice raw data into group tensors
        self._raw_group_data = [
            torch.tensor(self._data[:, list(group)], dtype=torch.float32, device=self.device) 
            for group in self._groups
        ]

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        # Not using mps due to potential non-implementation of certain operations in independence tests
        return torch.device('cpu')

    def extract_parents(self) -> dict[int, list[tuple[int, int]]]:
        if self._verbose > 0:
            logging.info("Extracting parents: Group-PCMCI with localized tests.")
        final_parents, _ = self._run_group_pcmci()
        return final_parents

    def _test_ci(self, x_var: int, x_lag: int, y_var: int, y_lag: int, z_list: list[tuple[int, int]]) -> tuple[float, float]:
        """
        Extracts time series data, applies regime masks, and delegates 
        the conditional independence testing to the statistical module.
        """
        T = self._raw_group_data[0].shape[0]
        start_t = 2 * self.tau_max 
        end_t = T
        
        if start_t >= end_t - 5:
            return 0.0, 1.0

        # 1. Extract target groups
        X_data = self._raw_group_data[x_var][start_t - x_lag : end_t - x_lag].to(torch.float32)
        Y_data = self._raw_group_data[y_var][start_t - y_lag : end_t - y_lag].to(torch.float32)
        
        # 2. Concatenate conditioning groups
        if z_list:
            Z_data_list = [self._raw_group_data[z_var][start_t - z_lag : end_t - z_lag].to(torch.float32) for z_var, z_lag in z_list]
            Z_data = torch.cat(Z_data_list, dim=1)
        else:
            Z_data = None

        # 3. Build regime masks
        if self.u is not None:
            u_sliced = self.u[start_t : end_t]
            num_regimes = u_sliced.shape[1]
            regime_masks = [u_sliced[:, r].bool() for r in range(num_regimes)]
        else:
            device = X_data.device if hasattr(self, 'device') else 'cpu'
            regime_masks = [torch.ones(end_t - start_t, dtype=torch.bool, device=device)]

        # 4. Chunk data by regime
        X_regimes, Y_regimes, Z_regimes = [], [], []
        for mask in regime_masks:
            if mask.sum().item() >= 6: # Minimum required for OLS and variance
                X_regimes.append(X_data[mask])
                Y_regimes.append(Y_data[mask])
                if Z_data is not None:
                    Z_regimes.append(Z_data[mask])

        if not X_regimes:
            return 0.0, 1.0

        # 5. Delegate to the independence test class (Pooled Residuals Early Fusion)
        if Z_data is not None:
            return self.conditional_independence_test.conditional_test_regimes(X_regimes, Y_regimes, Z_regimes)
        else:
            return self.conditional_independence_test.test_regimes(X_regimes, Y_regimes)

    def _run_group_pcmci(self) -> tuple[dict[int, list[tuple[int, int]]], list]:
        N = len(self._groups)
        independencies_found = []
        parents = {j: [(i, tau) for i in range(N) for tau in range(1, self.tau_max + 1)] for j in range(N)}
        
        # Phase 1: PC1 Algorithm
        for j in range(N):
            p = 0
            while p <= self.max_conds_dim:
                candidate_parents = list(parents[j])
                to_remove = [] 
                for (i, tau) in candidate_parents:
                    available_conds = [c for c in parents[j] if c != (i, tau)]
                    if len(available_conds) < p:
                        continue
                    for Z in itertools.combinations(available_conds, p):
                        _, pval = self._test_ci(i, tau, j, 0, list(Z))
                        if pval > self.pc_alpha:
                            to_remove.append((i, tau))
                            independencies_found.append((i, tau, j, 0, list(Z)))
                            break
                for node in to_remove:
                    if node in parents[j]:
                        parents[j].remove(node)
                p += 1

        # Phase 2: MCI Algorithm
        final_parents = {j: [] for j in range(N)}
        for j in range(N):
            for (i, tau) in parents[j]:
                Z_j = [c for c in parents[j] if c != (i, tau)]
                Z_i = [(k, tau_k + tau) for (k, tau_k) in parents[i]]
                Z = list(set(Z_j + Z_i))
                
                _, pval = self._test_ci(i, tau, j, 0, Z)
                if pval > self.pc_alpha:
                    independencies_found.append((i, tau, j, 0, Z))
                else:
                    final_parents[j].append((i, -tau))
        
        # Phase 3: Causal Input Completeness Augmentation
        if self._enforce_cic:
            independencies_found = self._enforce_causal_input_completeness(
                final_parents, independencies_found
            )
                    
        return final_parents, independencies_found

    def _build_time_indexed_graph(
        self, parents: dict[int, list[tuple[int, int]]]
    ) -> dict[tuple[int, int], set[tuple[int, int]]]:
        """
        Build the time-indexed graph from the discovered parents.
        
        Returns:
            adj: dict mapping (var, lag) → set of (parent_var, parent_lag) 
                 representing directed edges in the time-indexed graph.
        """
        N = len(self._groups)
        adj: dict[tuple[int, int], set[tuple[int, int]]] = {}
        
        # Initialize all nodes (variable, 0) for lag-0 variables
        for j in range(N):
            adj[(j, 0)] = set()
        
        # Add edges: (parent_var, -parent_lag) -> (child_var, 0)
        for j in range(N):
            for (i, tau) in parents.get(j, []):
                # Edge: X_i(tau) -> X_j(0), stored as (i, -tau) -> (j, 0)
                parent_node = (i, -tau)
                child_node = (j, 0)
                if parent_node not in adj:
                    adj[parent_node] = set()
                adj[child_node].add(parent_node)
        
        return adj

    def _compute_descendants(
        self, adj: dict[tuple[int, int], set[tuple[int, int]]]
    ) -> dict[tuple[int, int], set[tuple[int, int]]]:
        """
        Compute descendants for each node via BFS on the time-indexed graph.
        """
        descendants: dict[tuple[int, int], set[tuple[int, int]]] = {
            node: set() for node in adj
        }
        
        for start_node in adj:
            visited = set()
            queue = [start_node]
            while queue:
                current = queue.pop(0)
                for parent in adj.get(current, set()):
                    if parent not in visited:
                        visited.add(parent)
                        # parent is a parent of current, so start_node is an ancestor of parent
                        # We want descendants, so we reverse: if current has parent p,
                        # then current is a descendant of p
                        pass
                # Actually, we need to follow edges forward (from cause to effect)
                # adj[node] = set of parents of node
                # So to find descendants, we need to invert the adjacency
        
        # Build forward adjacency: node → set of children
        forward: dict[tuple[int, int], set[tuple[int, int]]] = {
            node: set() for node in adj
        }
        for node, parents_set in adj.items():
            for parent in parents_set:
                if parent in forward:
                    forward[parent].add(node)
        
        # BFS from each node following forward edges
        descendants = {node: set() for node in adj}
        for start_node in adj:
            visited = set()
            queue = [start_node]
            while queue:
                current = queue.pop(0)
                for child in forward.get(current, set()):
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)
            descendants[start_node] = visited
        
        return descendants

    def _enforce_causal_input_completeness(
        self,
        final_parents: dict[int, list[tuple[int, int]]],
        independencies_found: list,
    ) -> list:
        """
        Augment the independence statements to achieve causal input completeness.
        
        Tests the local Markov property CI statements that PCMCI may not have
        tested during edge pruning:
            X_j(0) ⊥ NonDesc(j) | Parents(j)
        
        For each variable j, we test whether j is independent of all its 
        non-descendants given its parents in the time-indexed graph.
        
        Args:
            final_parents: Discovered parent dict from PCMCI.
            independencies_found: List of CI statements already tested.
        
        Returns:
            Augmented list of independence statements.
        """
        N = len(self._groups)
        already_tested = set()
        for item in independencies_found:
            # item = (x_var, x_lag, y_var, y_lag, z_list)
            key = (item[0], item[1], item[2], item[3], tuple(tuple(z) for z in item[4]))
            already_tested.add(key)
        
        # Build time-indexed graph
        adj = self._build_time_indexed_graph(final_parents)
        descendants = self._compute_descendants(adj)
        
        augmented_count = 0
        
        for j in range(N):
            j_node = (j, 0)
            parents_j = adj.get(j_node, set())
            desc_j = descendants.get(j_node, set())
            
            # Non-descendants = all nodes except j and its descendants
            all_nodes = set(adj.keys())
            nondesc_j = all_nodes - desc_j - {j_node}
            
            # Build the conditioning set: parents of j at lag 0
            z_list = [(var, lag) for (var, lag) in parents_j]
            
            # Test: X_j(0) ⊥ NonDesc(j) | Parents(j)
            for (i, tau) in nondesc_j:
                # Skip if this was already tested during PCMCI
                test_key = (i, tau, j, 0, tuple(sorted(z_list)))
                reverse_key = (j, 0, i, tau, tuple(sorted(z_list)))
                
                if test_key in already_tested or reverse_key in already_tested:
                    continue
                
                # Test the CI statement
                _, pval = self._test_ci(i, tau, j, 0, z_list)
                
                if pval > self.pc_alpha:
                    # Independence holds — this is a new consistency statement
                    independencies_found.append((i, tau, j, 0, z_list))
                    already_tested.add(test_key)
                    augmented_count += 1
        
        if self._verbose > 0 and augmented_count > 0:
            logging.info(
                f"Causal input completeness: augmented {augmented_count} "
                f"local Markov CI statements."
            )
        
        return independencies_found
