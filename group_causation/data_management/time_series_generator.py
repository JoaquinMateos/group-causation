import numpy as np
import itertools
import math
from typing import Callable, Union, Tuple, List, Dict, cast

# Type alias for the causal links structure.
# Format: {node_j: [( ((parent1, lag1), (parent2, lag2), ...), coeff, func ), ...]}
CausalLinks = Dict[int, List[Tuple[Tuple[Tuple[int, int], ...], float, Callable[..., float]]]]


def _get_topological_order(links: CausalLinks, N: int) -> List[int]:
    """Computes the topological order of the contemporaneous (lag=0) sub-graph."""
    in_degree = {i: 0 for i in range(N)}
    adj_list = {i: [] for i in range(N)}

    for j in range(N):
        for parent_tuple, _, _ in links[j]:
            for parent_i, lag in parent_tuple:
                if lag == 0 and parent_i != j:
                    adj_list[parent_i].append(j)
                    in_degree[j] += 1

    queue = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []

    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for neighbor in adj_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) != N:
        raise ValueError("Cyclic contemporaneous links detected. DAG required for lag=0.")
    return topo_order


def _check_linear_stationarity(links: CausalLinks, N: int, max_lag: int) -> bool:
    """Checks stationarity by approximating the spectral radius of the companion matrix."""
    if max_lag == 0:
        return True

    graph = np.zeros((N, N, max_lag))
    for j in range(N):
        for parent_tuple, coeff, _ in links[j]:
            for parent_i, lag in parent_tuple:
                if lag < 0:
                    # Use absolute value of the coefficient to ensure we are checking the magnitude of influence, regardless of sign.
                    graph[j, parent_i, abs(lag) - 1] += abs(coeff)

    companion_dim = N * max_lag
    companion_matrix = np.zeros((companion_dim, companion_dim))
    
    for lag_idx in range(max_lag):
        companion_matrix[:N, lag_idx * N : (lag_idx + 1) * N] = graph[:, :, lag_idx]
        
    if max_lag > 1:
        companion_matrix[N:, :-N] = np.eye(N * (max_lag - 1))

    eigenvalues = np.linalg.eigvals(companion_matrix)
    return np.max(np.abs(eigenvalues)) < 1.0


def generate_group_causal_process_structure(
        groups: List[List[int]],
        group_links: Dict[int, List[Tuple[int, int]]], 
        n_node_links_per_group_link: int = 2,
        inner_group_density: float = 0.3,
        max_lag: int = 2,
        contemp_fraction: float = 0.0,
        cross_terms_fraction: float = 0.2, 
        dependency_funcs: List[Callable] = [lambda x: x], 
        multivariate_funcs: List[Callable] = [lambda x, y: x * y], 
        dependency_coeffs: List[float] = [-0.4, 0.4], 
        auto_coeffs: List[float] = [0.4], 
        seed: Union[int, None] = None,
        enforce_stationarity: bool = True
    ) -> CausalLinks:
    """
    Generates a node-level causal graph strictly derived from a predefined group-level structure.
    
    Args:
        groups: List of lists containing node IDs for each group. E.g. [[0, 1], [2, 3]]
        group_links: Macro-graph dictionary. Keys are target groups, values are lists of (parent_group, negative_lag).
                     E.g. {1: [(0, -1)]} means Group 0 causes Group 1 at lag -1.
    """
    rs = np.random.RandomState(seed)
    N = sum(len(g) for g in groups)
    max_tries = 100 if enforce_stationarity else 1
    
    for attempt in range(max_tries):
        incoming_edges = {i: [] for i in range(N)}

        # 1. GENERATE INTER GROUP LINKS
        for target_g, parents in group_links.items():
            for parent_g, neg_lag in parents:
                # Create multiple node-level links for each group-level link
                for _ in range(n_node_links_per_group_link):
                    node_t = rs.choice(groups[target_g])
                    node_p = rs.choice(groups[parent_g])
                    
                    if (node_p, neg_lag) not in incoming_edges[node_t]:
                        incoming_edges[node_t].append((node_p, neg_lag))

        # 2. GENERATE INTRA GROUP LINKS
        for g_idx, group_nodes in enumerate(groups):
            # Local topological order to prevent contemporaneous cycles within the group
            local_order = list(rs.permutation(group_nodes))
            for i, node_t in enumerate(local_order):
                for j, node_p in enumerate(local_order):
                    if i == j: 
                        continue
                    
                    if rs.rand() < inner_group_density:
                        # If node_p comes before node_t, allow contemporaneous connections with some probability
                        if j < i and rs.rand() < contemp_fraction:
                            neg_lag = 0
                        else:
                            neg_lag = -int(rs.randint(1, max_lag + 1)) if max_lag > 0 else 0
                            if neg_lag == 0: continue # Prevenir ciclos si i < j
                            
                        if (node_p, neg_lag) not in incoming_edges[node_t]:
                            incoming_edges[node_t].append((node_p, neg_lag))

        # 3. GENERATE AUTO-DEPENDENCIES
        links: CausalLinks = {i: [] for i in range(N)}
        if max_lag > 0 and auto_coeffs:
            for i in range(N):
                a_coeff = float(rs.choice(auto_coeffs))
                if a_coeff != 0.0:
                    links[i].append(( ((i, -1),), a_coeff, dependency_funcs[0] ))

        # 4. PACKAGE AND APPLY FUNCTIONS (Univariate / Multivariate) FOR CROSS-LINKS
        for target_node, parents in incoming_edges.items():
            rs.shuffle(parents)
            while parents:
                c = float(rs.choice(dependency_coeffs))
                if c == 0.0:
                    parents.pop()
                    continue
                
                # Create a multivariate term if there are at least 2 parents and it falls within the probability
                if len(parents) >= 2 and rs.rand() < cross_terms_fraction:
                    p1 = parents.pop()
                    p2 = parents.pop()
                    f = rs.choice(cast(list, multivariate_funcs))
                    links[target_node].append( ((p1, p2), c, f) )
                else:
                    p1 = parents.pop()
                    f = rs.choice(cast(list, dependency_funcs))
                    links[target_node].append( ((p1,), c, f) )

        # 5. CHECK STATIONARITY
        if not enforce_stationarity or _check_linear_stationarity(links, N, max_lag):
            return links
            
    raise ValueError("A stationary process could not be generated after 100 attempts. Reduce dependency_coeffs.")


def generate_data_from_causal_process_structure(
        links: CausalLinks, 
        T: int = 1000, 
        noise_dists: List[str] = ['gaussian'], 
        noise_sigmas: List[float] = [0.2], 
        transient_fraction: float = 0.2, 
        seed: Union[int, None] = None
    ) -> Tuple[np.ndarray, bool]:
    """Unrolls the equations over time to generate the synthetic dataset."""
    rs = np.random.RandomState(seed)
    N = len(links)

    max_lag = 0
    for j in range(N):
        for parent_tuple, _, _ in links[j]:
            for _, lag in parent_tuple:
                max_lag = max(max_lag, abs(lag))

    transient = int(math.floor(transient_fraction * T))
    total_T = T + transient

    data = np.zeros((total_T, N), dtype='float32')
    for j in range(N):
        dist = rs.choice(noise_dists)
        sigma = rs.choice(noise_sigmas)
        
        if dist == 'gaussian': data[:, j] = rs.normal(0, sigma, total_T)
        elif dist == 'uniform': data[:, j] = rs.uniform(-sigma, sigma, total_T)
        elif dist == 'weibull':
            a = 2.0
            mean_w, var_w = math.gamma(1.5), math.gamma(2.0) - math.gamma(1.5)**2
            data[:, j] = sigma * (rs.weibull(a, total_T) - mean_w) / np.sqrt(var_w)

    causal_order = _get_topological_order(links, N)

    for t in range(max_lag, total_T):
        for j in causal_order:
            for parent_tuple, coeff, func in links[j]:
                # Extraer los valores de todos los padres involucrados en este término
                parent_vals = [data[t + lag, p_i] for p_i, lag in parent_tuple]
                data[t, j] += coeff * func(*parent_vals) # Multivariate unpacking

    data_final = data[transient:]
    nonvalid = bool(np.any(np.isnan(data_final)) or np.any(np.isinf(data_final)))

    return data_final, nonvalid

# ==============================================================================
# EXAMPLE OF USE
# ==============================================================================
if __name__ == '__main__':
    
    # 1. Definimos los Grupos
    grupos_definidos = [[0, 1, 2], [3, 4], [5, 6, 7]]
    
    # 2. Definimos el Macro-Grafo causal (Enlaces entre grupos)
    # Por ejemplo: el Grupo 0 causa al Grupo 1 en lag -1. El Grupo 1 causa al Grupo 2 en lag -2.
    enlaces_entre_grupos = {
        1: [(0, -1)],
        2: [(1, -2)],
    }
    
    # 3. Generamos la estructura a nivel de Nodo (Micro-Grafo)
    estructura_nodos = generate_group_causal_process_structure(
        groups=grupos_definidos,
        group_links=enlaces_entre_grupos,
        n_node_links_per_group_link=2, # Crea 2 flechas entre nodos por cada enlace de grupo
        inner_group_density=0.4,       # Ruido causal dentro del propio grupo
        dependency_funcs=[lambda x: x, np.sin],
        multivariate_funcs=[lambda x, y: x * y],
        dependency_coeffs=[-0.3, 0.3],
        seed=42
    )
    
    # 4. Generamos las series temporales
    time_series, has_nans = generate_data_from_causal_process_structure(
        links=estructura_nodos,
        T=2000,
        noise_sigmas=[0.2]
    )
    
    print(f"Shape de las series temporales generadas: {time_series.shape}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for i in range(time_series.shape[1]):
        plt.plot(time_series[:, i] + i*5, label=f'Node {i}')  # Desplazamos cada nodo para visualización
    plt.title('Series Temporales Sintéticas con Estructura Causal de Grupos')
    plt.xlabel('Time')
    plt.ylabel('Value (offset for visibility)')
    plt.legend()
    plt.show()