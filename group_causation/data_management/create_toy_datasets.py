"""Toy dataset generation and persistence for causal time series."""

import ast
import logging
import os
import random
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
from tigramite import plotting as tp
from tigramite.graphs import Graphs
from tigramite.toymodels.structural_causal_processes import (
    generate_structural_causal_process,
    structural_causal_process,
)

from group_causation.data_management.latent_macro_scm import LatentMacroDataset, generate_latent_macro_scm
from group_causation.data_management.time_series_generator import (
    generate_group_causal_process_structure,
    generate_data_from_causal_process_structure,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def get_parents_dict(causal_process) -> dict[int, list[tuple[int, int]]]:
    """Extract a parents dict from a Tigramite or multivariate causal process."""
    parents_dict: dict[int, list[tuple[int, int]]] = {}
    for key, links in causal_process.items():
        parents_dict[key] = []
        for link in links:
            parents_info = link[0]
            # Multivariate format: ((parent, lag), ...) tuple of tuples
            if isinstance(parents_info[0], tuple):
                for p, lag in parents_info:
                    if (p, lag) not in parents_dict[key]:
                        parents_dict[key].append((p, lag))
            # Tigramite format: ((parent, lag), coeff, func)
            else:
                p, lag = parents_info
                if (p, lag) not in parents_dict[key]:
                    parents_dict[key].append((p, lag))
    return parents_dict


def _extract_subgraph(parents: dict[int, list[tuple[int, int]]],
                      chosen_nodes: list[int]) -> dict[int, list[tuple[int, int]]]:
    """Return the subgraph induced by *chosen_nodes*.

    A variable in *chosen_nodes* is a child of another iff there is a
    directed path from the parent to the child through non-chosen nodes.
    """
    chosen_set = set(chosen_nodes)
    idx_of = {node: i for i, node in enumerate(chosen_nodes)}

    new_parents: dict[int, list[tuple[int, int]]] = {idx: [] for idx in idx_of.values()}

    for child in chosen_nodes:
        child_idx = idx_of[child]
        queue: deque[tuple[int, int]] = deque([(child, 0)])
        visited: set[int] = {child}

        while queue:
            curr, cum_lag = queue.popleft()

            for p, lag in parents.get(curr, []):
                # Keep direct autoregressive links of the chosen node.
                if curr == child and p == child and lag < 0:
                    edge = (idx_of[p], cum_lag + lag)
                    if edge not in new_parents[child_idx]:
                        new_parents[child_idx].append(edge)
                    continue

                if p in visited:
                    continue
                total_lag = cum_lag + lag

                if p in chosen_set:
                    if not (p == child and lag == 0):
                        new_parents[child_idx].append((idx_of[p], total_lag))
                else:
                    visited.add(p)
                    queue.append((p, total_lag))

        # Deduplicate
        seen: set[tuple[int, int]] = set()
        uniq: list[tuple[int, int]] = []
        for pair in new_parents[child_idx]:
            if pair not in seen:
                seen.add(pair)
                uniq.append(pair)
        new_parents[child_idx] = uniq

    return new_parents


# ---------------------------------------------------------------------------
# Persistence mixin (safe loading, no eval)
# ---------------------------------------------------------------------------

class DatasetPersistence:
    """Save / load helpers for CausalDataset files."""

    def _save(self, name: str, dataset_folder: str) -> None:
        """Save time_series and parents_dict to CSV + TXT."""
        df = pd.DataFrame(self.time_series)
        df.to_csv(f"{dataset_folder}/{name}_data.csv", index=False, header=True)
        with open(f"{dataset_folder}/{name}_parents.txt", "w") as f:
            f.write(repr(self.parents_dict))

    def _save_groups(self, name: str, dataset_folder: str) -> None:
        """Save time_series, parents, groups, node_parents, NS info, and latent ground truth."""
        self._save(name, dataset_folder)
        with open(f"{dataset_folder}/{name}_groups.txt", "w") as f:
            f.write(repr(self._groups))
        with open(f"{dataset_folder}/{name}_node_parents.txt", "w") as f:
            f.write(repr(self.node_parents_dict))
        with open(f"{dataset_folder}/{name}_non_stationarity_info.txt", "w") as f:
            f.write(repr(self.non_stationarity_info))
        if self.latent_true is not None:
            np.savetxt(f"{dataset_folder}/{name}_latent_true.csv", self.latent_true, delimiter=",")
        if self.latent_dims is not None:
            with open(f"{dataset_folder}/{name}_latent_dims.txt", "w") as f:
                f.write(repr(self.latent_dims))
        if self.u is not None:
            np.savetxt(f"{dataset_folder}/{name}_u.csv", self.u, delimiter=",")

    @staticmethod
    def load_literal(filepath: str) -> Any:
        """Safely load a repr-written Python literal from a text file."""
        with open(filepath) as f:
            return ast.literal_eval(f.read())

    @classmethod
    def load_parents_dict(cls, filepath: str) -> dict[int, list[tuple[int, int]]]:
        """Safely load a parents dict from a repr-written text file."""
        return cls.load_literal(filepath)

    @classmethod
    def load_groups(cls, filepath: str) -> list[list[int]]:
        """Safely load groups from a repr-written text file."""
        return cls.load_literal(filepath)

    @classmethod
    def load_node_parents_dict(cls, filepath: str) -> dict[int, list[tuple[int, int]]]:
        """Safely load a node parents dict from a repr-written text file."""
        return cls.load_literal(filepath)

    @classmethod
    def load_non_stationarity_info(cls, filepath: str) -> dict[str, Any]:
        """Safely load non-stationarity info from a repr-written text file."""
        return cls.load_literal(filepath)

    @staticmethod
    def load_array(filepath: str) -> np.ndarray:
        """Load a 2D numeric array saved with ``np.savetxt``."""
        return np.loadtxt(filepath, delimiter=",", ndmin=2)


# ---------------------------------------------------------------------------
# Graph-transform helpers
# ---------------------------------------------------------------------------

def extract_group_parents(node_parents_dict: dict[int, list[tuple[int, int]]],
                          groups: list[list[int]]) -> dict[int, list[tuple[int, int]]]:
    """Map node-level parents to group-level parents.

    A group is a parent of another group iff any node in the child group
    has a parent node that belongs to the parent group.
    """
    # Pre-build node → group index for O(1) lookups
    node_to_group: dict[int, int] = {}
    for group_idx, group in enumerate(groups):
        for node in group:
            node_to_group[node] = group_idx

    n_groups = len(groups)
    group_parents_dict: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_groups)}

    for son_node, parents in node_parents_dict.items():
        son_group = node_to_group[son_node]
        for parent, lag in parents:
            parent_group = node_to_group[parent]
            group_parents_dict[son_group].append((parent_group, lag))

        # Deduplicate and remove self-loops (lag 0)
        group_parents_dict[son_group] = [
            (p, lag) for p, lag in set(group_parents_dict[son_group])
            if not (p == son_group and lag == 0)
        ]

    return group_parents_dict


def generate_groups(n_vars: int, n_groups: int) -> list[list[int]]:
    """Generate *n_groups* groups with at least 2 nodes each."""
    if n_groups > n_vars / 2:
        raise ValueError("The number of groups must be less than N_vars / 2")

    nodes = list(range(n_vars))
    groups = [[nodes.pop(), nodes.pop()] for _ in range(n_groups)]

    while nodes:
        groups[random.randint(0, n_groups - 1)].append(nodes.pop())

    return groups


def generate_random_group_links(n_groups: int, density: float, max_lag: int,
                                contemp_fraction: float) -> dict[int, list[tuple[int, int]]]:
    """Generate a random macro-graph (group-level links)."""
    group_links: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_groups)}
    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                continue
            if random.random() < density:
                if j > i and random.random() < contemp_fraction:
                    lag = 0
                else:
                    lag = -random.randint(1, max_lag) if max_lag > 0 else 0
                    if lag == 0:
                        continue

                if (i, lag) not in group_links[j]:
                    group_links[j].append((i, lag))
    return group_links


# ---------------------------------------------------------------------------
# Dependency functions registry
# ---------------------------------------------------------------------------

DEPENDENCY_FUNCS: dict[str, Any] = {
    "linear": lambda x: x,
    "negative-exponential": lambda x: 1 - np.exp(-abs(x)),
    "sin": lambda x: np.sin(x),
    "cos": lambda x: np.cos(x),
    "step": lambda x: 1 if x > 0 else -1,
}


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class CausalDataset(DatasetPersistence):
    """Container for causal time series data and its ground-truth graph.

    Can be initialised empty and populated via ``generate_toy_data`` or
    ``generate_group_toy_data``, or constructed directly from arrays.
    """

    def __init__(self, time_series=None, parents_dict=None, groups=None,
                 max_value_threshold: float = 1e10):
        self.time_series: np.ndarray | None = time_series
        self.parents_dict: dict[int, list[tuple[int, int]]] | None = parents_dict
        self._groups: list[list[int]] | None = groups
        self.node_parents_dict: dict[int, list[tuple[int, int]]] = {}
        self.max_value_threshold = max_value_threshold
        self.non_stationarity_info: dict[str, bool | list[int] | list[float]] = {"applied": False}
        self.latent_true: np.ndarray | None = None
        self.latent_dims: list[int] | None = None
        self.u: np.ndarray | None = None
        self.latent_macro_case: str | None = None

    @property
    def groups(self) -> list[list[int]] | None:
        return self._groups

    @groups.setter
    def groups(self, value: list[list[int]] | None) -> None:
        self._groups = value

    # ------------------------------------------------------------------
    # Node-level toy data generation
    # ------------------------------------------------------------------

    def generate_toy_data(self, name: str, T: int = 100, N_vars: int = 10,
                          crosslinks_density: float = 0.75,
                          confounders_density: float = 0.0,
                          min_lag: int = 1, max_lag: int = 3,
                          contemp_fraction: float = 0.0,
                          dependency_funcs=None,
                          datasets_folder: str | None = None,
                          maximum_tries: int = 100,
                          **kw_generation_args) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
        """Generate a node-level toy dataset from a random causal process."""
        if min_lag > 0 and contemp_fraction > 1e-6:
            raise ValueError("If min_lag > 0, then contemp_fraction must be 0")
        if min_lag == 0 and contemp_fraction < 1e-6:
            raise ValueError("If min_lag is 0, then contemp_fraction can not be 0.")

        if dependency_funcs is None:
            dependency_funcs = ["nonlinear"]
        parsed_funcs = [DEPENDENCY_FUNCS.get(f, f) for f in dependency_funcs]

        L = int((N_vars * crosslinks_density / (1 - crosslinks_density)) // (1 - contemp_fraction))
        total_generating_vars = int(N_vars * (1 + confounders_density))

        for it in range(1, maximum_tries + 1):
            causal_process, noise = generate_structural_causal_process(
                N=total_generating_vars, L=L, max_lag=max_lag,
                contemp_fraction=contemp_fraction,
                dependency_funcs=parsed_funcs, **kw_generation_args,
            )
            self.parents_dict = get_parents_dict(causal_process)
            self.time_series, _ = structural_causal_process(causal_process, T=T, noises=noise)

            if confounders_density > 1e-6:
                chosen_nodes = random.sample(range(total_generating_vars), N_vars)
                self.time_series = self.time_series[:, chosen_nodes]
                self.parents_dict = _extract_subgraph(self.parents_dict, chosen_nodes)

            if (np.all(np.isfinite(self.time_series))
                    and np.all(np.abs(self.time_series) < self.max_value_threshold)
                    and not np.any(np.isnan(self.time_series))):
                break
            logger.debug("Dataset has NaNs or infinites, retrying... %d/%d", it, maximum_tries)
        else:
            raise ValueError("Could not generate a dataset without NaNs")

        if datasets_folder is not None:
            os.makedirs(datasets_folder, exist_ok=True)
            self._save(name, datasets_folder)

        assert self.time_series is not None
        assert self.parents_dict is not None
        return self.time_series, self.parents_dict

    # ------------------------------------------------------------------
    # Group-level toy data generation
    # ------------------------------------------------------------------

    def generate_group_toy_data(self, name: str, T: int = 100, N_vars: int = 20,
                                N_groups: int = 3,
                                inner_group_crosslinks_density: float = 0.5,
                                outer_group_crosslinks_density: float = 0.5,
                                latent_confounding_fraction: float = 0.0,
                                maximum_of_nodes_confounded: int = 4,
                                n_node_links_per_group_link: int = 2,
                                contemp_fraction: float = 0.0,
                                cross_terms_fraction: float = 0.2,
                                max_lag: int = 3, min_lag: int = 1,
                                dependency_funcs=None,
                                multivariate_funcs=None,
                                dependency_coeffs=None,
                                auto_coeffs=None,
                                noise_dists=None,
                                noise_sigmas=None,
                                datasets_folder: str | None = None,
                                maximum_tries: int = 100,
                                group_links=None,
                                non_stationarity_params=None,
                                **kw_generation_args):
        """Generate a group-level toy dataset with latent confounding support."""
        if min_lag > 0 and contemp_fraction > 1e-6:
            raise ValueError("If there is a fraction of contemporaneous links, min_lag must be 0")

        # Defaults
        if dependency_funcs is None:
            dependency_funcs = ["linear"]
        if multivariate_funcs is None:
            multivariate_funcs = [lambda x, y: x * y]
        if dependency_coeffs is None:
            dependency_coeffs = [-0.5, 0.5]
        if auto_coeffs is None:
            auto_coeffs = [0.5, 0.7]
        if noise_dists is None:
            noise_dists = ["gaussian"]
        if noise_sigmas is None:
            noise_sigmas = [0.5, 2]
        if non_stationarity_params is None:
            non_stationarity_params = {}

        parsed_funcs = [DEPENDENCY_FUNCS.get(f, f) for f in dependency_funcs]
        total_vars = int(N_vars * (1 + latent_confounding_fraction))

        for it in range(1, maximum_tries + 1):
            try:
                current_groups = generate_groups(total_vars, N_groups)

                if group_links is None:
                    current_group_links = generate_random_group_links(
                        N_groups, outer_group_crosslinks_density, max_lag, contemp_fraction
                    )
                else:
                    current_group_links = group_links

                global_causal_process, latent_nodes = generate_group_causal_process_structure(
                    groups=current_groups,
                    group_links=current_group_links,
                    n_node_links_per_group_link=n_node_links_per_group_link,
                    inner_group_density=inner_group_crosslinks_density,
                    latent_confounding_fraction=latent_confounding_fraction,
                    max_lag=max_lag,
                    contemp_fraction=contemp_fraction,
                    cross_terms_fraction=cross_terms_fraction,
                    dependency_funcs=parsed_funcs,
                    multivariate_funcs=multivariate_funcs,
                    dependency_coeffs=dependency_coeffs,
                    auto_coeffs=auto_coeffs,
                    enforce_stationarity=(non_stationarity_params != {}),
                )

                visible_nodes = [n for n in range(total_vars) if n not in latent_nodes]

                full_time_series, nonvalid, self.non_stationarity_info = (
                    generate_data_from_causal_process_structure(
                        links=global_causal_process, T=T,
                        noise_dists=noise_dists, noise_sigmas=noise_sigmas,
                        non_stationarity_params=non_stationarity_params,
                    )
                )
                self.stationarity_info = self.non_stationarity_info

                if nonvalid or np.any(np.abs(full_time_series) > self.max_value_threshold):
                    continue

                self.time_series = full_time_series[:, visible_nodes]

                full_node_parents = get_parents_dict(global_causal_process)
                self.node_parents_dict = _extract_subgraph(full_node_parents, visible_nodes)

                # Remap group indices to contiguous 0..N_vars-1
                node_mapping = {old: new for new, old in enumerate(visible_nodes)}
                self._groups = []
                for g in current_groups:
                    new_g = [node_mapping[n] for n in g if n in node_mapping]
                    if new_g:
                        self._groups.append(new_g)

                self.parents_dict = extract_group_parents(self.node_parents_dict, self._groups)
                break

            except Exception as e:
                logger.exception("Generation attempt %d/%d failed: %s", it, maximum_tries, e)
                if it == maximum_tries:
                    raise ValueError(
                        f"Could not generate a dataset after {maximum_tries} tries. Last error: {e}"
                    )

        if datasets_folder is not None:
            os.makedirs(datasets_folder, exist_ok=True)
            self._save_groups(name, datasets_folder)

        assert self.time_series is not None
        assert self.parents_dict is not None
        assert self._groups is not None

        return self.time_series, self.parents_dict, self._groups, self.node_parents_dict, self.non_stationarity_info

    # ------------------------------------------------------------------
    # Latent-macro SCM generation
    # ------------------------------------------------------------------

    def generate_latent_macro_data(self, name: str, datasets_folder: str | None = None,
                                   **scm_params) -> LatentMacroDataset:
        """Generate a latent-macro SCM dataset preserving the ground-truth latents.

        See :func:`generate_latent_macro_scm` for the accepted arguments.
        """
        dataset = generate_latent_macro_scm(**scm_params)
        self._apply_latent_macro_dataset(dataset)

        if datasets_folder is not None:
            os.makedirs(datasets_folder, exist_ok=True)
            self._save_groups(name, datasets_folder)

        return dataset

    def _apply_latent_macro_dataset(self, dataset: LatentMacroDataset) -> None:
        self.time_series = dataset.time_series
        self.parents_dict = dataset.group_parents
        self._groups = dataset.groups
        self.node_parents_dict = {}
        self.non_stationarity_info = dataset.non_stationarity_info
        self.latent_true = dataset.latent_true
        self.latent_dims = dataset.latent_dims
        self.u = dataset.u
        self.latent_macro_case = dataset.case

    # ------------------------------------------------------------------
    # Delegation to module-level function (backward compat)
    # ------------------------------------------------------------------

    def extract_group_parents(self, node_parents_dict: dict[int, list[tuple[int, int]]]) -> dict[int, list[tuple[int, int]]]:
        """Delegate to the module-level function using this instance's groups."""
        assert self._groups is not None
        return extract_group_parents(node_parents_dict, self._groups)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ts_graph(parents_dict, var_names=None):
    """Plot the graph structure of a time series causal process."""
    graph = Graphs.get_graph_from_dict(parents_dict)
    tp.plot_time_series_graph(
        graph=graph,
        var_names=var_names,
        link_colorbar_label="cross-MCI (edges)",
    )
