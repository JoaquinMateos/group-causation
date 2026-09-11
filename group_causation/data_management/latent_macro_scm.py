"""Synthetic SCMs for aggregation-faithfulness research.

This module builds datasets where a small set of macro-latent variables
``Z*`` follow a causal process and a per-group, injective, non-linear
mixing function maps them to high-dimensional micro-variables ``X``::

    Z*_t = f_causal(Z*_{<t}, eta_t, u)
    X_t^g = f_mix^g(Z*_t^g) + epsilon_t^g

Unlike :func:`CausalDataset.generate_group_toy_data`, the macro-latents are
kept as ground truth, so latent-identifiability metrics (MCC) can be computed.

The four ``CASE_PRESETS`` reproduce the failure modes where PCA-based
aggregation breaks but identifiable VAE aggregation succeeds.
"""

import math
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np

from group_causation.data_management.time_series_generator import (
    generate_data_from_causal_process_structure,
    generate_group_causal_process_structure,
)


class LatentMacroDataset(NamedTuple):
    """Ground-truth dataset with observable micro-variables and latent macro-variables."""

    time_series: np.ndarray
    latent_true: np.ndarray
    latent_dims: list[int]
    groups: list[list[int]]
    group_parents: dict[int, list[tuple[int, int]]]
    u: np.ndarray | None
    non_stationarity_info: dict[str, Any]
    case: str | None


ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'tanh': np.tanh,
    'sin': np.sin,
    'cos': np.cos,
    'relu': lambda x: np.maximum(x, 0.0),
}

def _regime_shifts(num_shifts: int = 4, max_mean_mod: float = 2.0, max_std_mod: float = 3.0) -> dict[str, Any]:
    """Environmental contexts that modulate the latent distribution.

    iVAE identifiability requires the auxiliary variable ``u`` to be
    informative, so every failure case includes a baseline of regime shifts
    unless the caller overrides ``non_stationarity_params``.
    """
    return {
        'type': 'regime_shifts',
        'fraction': 1.0,
        'num_shifts': num_shifts,
        'max_mean_mod': max_mean_mod,
        'max_std_mod': max_std_mod,
    }


CASE_PRESETS: dict[str, dict[str, Any]] = {
    'nonlinear_mixing': {
        'mixing_activation': 'tanh',
        'observation_noise_scale': 0.1,
        'latent_dim_per_group': 3,
        'non_stationarity_params': _regime_shifts(),
    },
    'low_variance_causal': {
        'causal_signal_scale': 0.05,
        'observation_noise_scale': 2.0,
        'non_stationarity_params': _regime_shifts(),
    },
    'nonstationary_regimes': {
        'non_stationarity_params': _regime_shifts(num_shifts=6, max_mean_mod=4.0, max_std_mod=3.0),
    },
    'nongaussian_innovations': {
        'noise_dists': ('laplace',),
        'noise_sigmas': (0.5,),
        'non_stationarity_params': _regime_shifts(),
    },
}

_DEFAULT_OPTIONS: dict[str, Any] = {
    'dependency_funcs': (np.tanh,),
    'dependency_coeffs': (-0.6, 0.6),
    'auto_coeffs': (0.5,),
    'noise_dists': ('gaussian',),
    'noise_sigmas': (0.5,),
    'noise_dist_params': None,
    'causal_signal_scale': 1.0,
    'observation_noise_scale': 0.1,
    'mixing_activation': 'tanh',
    'latent_dim_per_group': 1,
    'non_stationarity_params': None,
}


def generate_latent_macro_scm(
        n_groups: int,
        micro_dim: int | Sequence[int],
        T: int = 1000,
        *,
        case: str | None = None,
        group_links: dict[int, list[tuple[int, int]]] | None = None,
        latent_dim_per_group: int | None = None,
        dependency_funcs: Sequence[Callable] | None = None,
        dependency_coeffs: Sequence[float] | None = None,
        auto_coeffs: Sequence[float] | None = None,
        noise_dists: Sequence[str] | None = None,
        noise_sigmas: Sequence[float] | None = None,
        noise_dist_params: dict[str, dict[str, float]] | None = None,
        causal_signal_scale: float | None = None,
        observation_noise_scale: float | None = None,
        mixing_activation: str | Callable | None = None,
        mixing_hidden_dim: int = 16,
        mixing_n_hidden_layers: int = 2,
        max_lag: int = 2,
        non_stationarity_params: dict[str, Any] | None = None,
        transient_fraction: float = 0.2,
        seed: int | None = None,
) -> LatentMacroDataset:
    """Generate a latent-macro SCM dataset for aggregation-faithfulness experiments.

    Args:
        n_groups: Number of macro-groups (latent causal variables).
        micro_dim: Micro-variables per group. Either a single int (same for all
            groups) or a sequence of length ``n_groups``.
        T: Number of post-transient time steps.
        case: Optional name of a preset from ``CASE_PRESETS``. Explicit keyword
            arguments always override preset values.
        group_links: Macro causal graph ``{child: [(parent, lag), ...]}`` with
            negative lags. Defaults to a deterministic chain.
        latent_dim_per_group: Latent causal dimensions per group. Values above
            ``1`` create the curved manifolds that linear projections cannot
            unmix.
        seed: Random seed for reproducibility.
        transient_fraction: Fraction of samples discarded as warm-up.

    Returns:
        A :class:`LatentMacroDataset` with micro time series, ground-truth
        latents, groups, macro graph, regime labels ``u`` and non-stationarity info.
    """
    if case is not None and case not in CASE_PRESETS:
        raise ValueError(f"Unknown case: {case}. Options: {list(CASE_PRESETS)}")

    options = _resolve_options(case, {
        'dependency_funcs': dependency_funcs,
        'dependency_coeffs': dependency_coeffs,
        'auto_coeffs': auto_coeffs,
        'noise_dists': noise_dists,
        'noise_sigmas': noise_sigmas,
        'noise_dist_params': noise_dist_params,
        'causal_signal_scale': causal_signal_scale,
        'observation_noise_scale': observation_noise_scale,
        'mixing_activation': mixing_activation,
        'latent_dim_per_group': latent_dim_per_group,
        'non_stationarity_params': non_stationarity_params,
    })
    micro_dims = _resolve_micro_dims(n_groups, micro_dim)
    latent_dims = _resolve_latent_dims(n_groups, options['latent_dim_per_group'], micro_dims)
    groups = _build_groups(micro_dims)
    macro_links = group_links if group_links is not None else _default_macro_links(n_groups, max_lag)

    latent_structure = _build_latent_structure(
        n_groups=n_groups,
        latent_dims=latent_dims,
        group_links=macro_links,
        max_lag=max_lag,
        dependency_funcs=options['dependency_funcs'],
        dependency_coeffs=options['dependency_coeffs'],
        auto_coeffs=options['auto_coeffs'],
        seed=seed,
    )
    latent_data, nonvalid, non_stationarity_info = generate_data_from_causal_process_structure(
        links=latent_structure,
        T=T,
        noise_dists=list(options['noise_dists']),
        noise_sigmas=[sigma * options['causal_signal_scale'] for sigma in options['noise_sigmas']],
        noise_dist_params=options['noise_dist_params'],
        transient_fraction=transient_fraction,
        seed=seed,
        non_stationarity_params=options['non_stationarity_params'] or {},
    )
    if nonvalid:
        raise ValueError('Latent process generated invalid values. Reduce dependency_coeffs.')

    time_series = _mix_latents_into_micro(
        latent_data=latent_data,
        latent_dims=latent_dims,
        micro_dims=micro_dims,
        activation=_resolve_activation(options['mixing_activation']),
        hidden_dim=mixing_hidden_dim,
        n_hidden_layers=mixing_n_hidden_layers,
        noise_scale=options['observation_noise_scale'],
        seed=seed,
    )

    return LatentMacroDataset(
        time_series=time_series,
        latent_true=latent_data,
        latent_dims=latent_dims,
        groups=groups,
        group_parents=_macro_parents_from_links(latent_structure, latent_dims),
        u=_build_regime_one_hot(non_stationarity_info, T),
        non_stationarity_info=non_stationarity_info,
        case=case,
    )


def _resolve_options(case: str | None, explicit: dict[str, Any]) -> dict[str, Any]:
    """Merge defaults, case preset and explicit overrides (last one wins)."""
    preset = CASE_PRESETS.get(case, {}) if case else {}
    provided = {name: value for name, value in explicit.items() if value is not None}
    return {**_DEFAULT_OPTIONS, **preset, **provided}


def _resolve_micro_dims(n_groups: int, micro_dim: int | Sequence[int]) -> list[int]:
    dims = [int(micro_dim)] * n_groups if isinstance(micro_dim, (int, np.integer)) else [int(d) for d in micro_dim]
    if len(dims) != n_groups:
        raise ValueError(f'Expected {n_groups} micro dimensions. Got {len(dims)}.')
    if any(dim < 1 for dim in dims):
        raise ValueError(f'Micro dimensions must be >= 1. Got {dims}.')
    return dims


def _resolve_latent_dims(n_groups: int, latent_dim_per_group: int, micro_dims: Sequence[int]) -> list[int]:
    latent_dims = [int(latent_dim_per_group)] * n_groups
    if any(dim < 1 for dim in latent_dims):
        raise ValueError(f'latent_dim_per_group must be >= 1. Got {latent_dim_per_group}.')
    for dim, micro_dim in zip(latent_dims, micro_dims):
        if dim > micro_dim:
            raise ValueError(
                f'latent_dim_per_group ({dim}) cannot exceed the group micro_dim ({micro_dim}); '
                'the mixing map needs at least as many outputs as latent sources.'
            )
    return latent_dims


def _build_groups(micro_dims: Sequence[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    start = 0
    for dim in micro_dims:
        groups.append(list(range(start, start + dim)))
        start += dim
    return groups


def _default_macro_links(n_groups: int, max_lag: int) -> dict[int, list[tuple[int, int]]]:
    """Deterministic chain graph with one skip edge to break linear-chain symmetry."""
    links: dict[int, list[tuple[int, int]]] = {group: [] for group in range(n_groups)}
    for group in range(1, n_groups):
        links[group].append((group - 1, -1))
    if n_groups > 2 and max_lag >= 2:
        links[2].append((0, -2))
    return links


def _build_latent_structure(
        n_groups: int,
        latent_dims: Sequence[int],
        group_links: dict[int, list[tuple[int, int]]],
        max_lag: int,
        dependency_funcs: Sequence[Callable],
        dependency_coeffs: Sequence[float],
        auto_coeffs: Sequence[float],
        seed: int | None,
) -> dict:
    structure, _ = generate_group_causal_process_structure(
        groups=_build_groups(latent_dims),
        group_links=group_links,
        n_node_links_per_group_link=1,
        inner_group_density=0.3,
        max_lag=max_lag,
        contemp_fraction=0.0,
        dependency_funcs=list(dependency_funcs),
        dependency_coeffs=list(dependency_coeffs),
        auto_coeffs=list(auto_coeffs),
        enforce_autoregression=True,
        seed=seed,
        enforce_stationarity=True,
    )
    return structure


def _macro_parents_from_links(latent_structure: dict, latent_dims: Sequence[int]) -> dict[int, list[tuple[int, int]]]:
    """Collapse latent-level links into the group-level ground-truth graph."""
    node_to_group = {
        node: group
        for group, nodes in enumerate(_build_groups(latent_dims))
        for node in nodes
    }
    parents: dict[int, list[tuple[int, int]]] = {group: [] for group in range(len(latent_dims))}
    for child, terms in latent_structure.items():
        child_group = node_to_group[child]
        for parent_tuple, _, _ in terms:
            for parent, lag in parent_tuple:
                edge = (node_to_group[parent], lag)
                if edge == (child_group, 0):
                    continue
                if edge not in parents[child_group]:
                    parents[child_group].append(edge)
    return parents


def _resolve_activation(activation: str | Callable | None) -> Callable[[np.ndarray], np.ndarray]:
    if activation is None:
        return np.tanh
    if callable(activation):
        return activation
    if activation not in ACTIVATIONS:
        raise ValueError(f'Unknown mixing activation: {activation}. Options: {list(ACTIVATIONS)}')
    return ACTIVATIONS[activation]


def _build_mixing_layers(input_dim: int, output_dim: int, hidden_dim: int,
                         n_hidden_layers: int, rs: np.random.RandomState) -> list[tuple[np.ndarray, np.ndarray]]:
    dims = [input_dim] + [hidden_dim] * n_hidden_layers + [output_dim]
    return [
        (rs.normal(0.0, 1.0 / math.sqrt(fan_in), size=(fan_in, fan_out)), rs.normal(0.0, 0.1, size=fan_out))
        for fan_in, fan_out in zip(dims[:-1], dims[1:])
    ]


def _apply_mixing(latent: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]],
                 activation: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    hidden = latent
    for layer_index, (weight, bias) in enumerate(layers):
        hidden = hidden @ weight + bias
        if layer_index < len(layers) - 1:
            hidden = activation(hidden)
    return hidden


def _mix_latents_into_micro(
        latent_data: np.ndarray,
        latent_dims: Sequence[int],
        micro_dims: Sequence[int],
        activation: Callable[[np.ndarray], np.ndarray],
        hidden_dim: int,
        n_hidden_layers: int,
        noise_scale: float,
        seed: int | None,
) -> np.ndarray:
    """Apply one injective non-linear mixing MLP plus observation noise per group."""
    mixing_rs = np.random.RandomState(None if seed is None else seed + 1)
    time_series = np.zeros((latent_data.shape[0], sum(micro_dims)), dtype=np.float64)

    start_micro = 0
    start_latent = 0
    for group, dim in enumerate(micro_dims):
        n_latent = latent_dims[group]
        layers = _build_mixing_layers(
            input_dim=n_latent,
            output_dim=dim,
            hidden_dim=max(hidden_dim, 2 * dim),
            n_hidden_layers=n_hidden_layers,
            rs=mixing_rs,
        )
        mixed = _apply_mixing(latent_data[:, start_latent:start_latent + n_latent], layers, activation)
        time_series[:, start_micro:start_micro + dim] = (
            mixed + mixing_rs.normal(0.0, noise_scale, size=(latent_data.shape[0], dim))
        )
        start_micro += dim
        start_latent += n_latent

    return time_series


def _build_regime_one_hot(non_stationarity_info: dict[str, Any], T: int) -> np.ndarray | None:
    """Build one-hot regime labels from regime-shift metadata, if present."""
    if not non_stationarity_info.get('applied') or non_stationarity_info.get('type') != 'regime_shifts':
        return None

    affected_vars = non_stationarity_info.get('affected_vars', [])
    shift_details = non_stationarity_info.get('shift_details', {})
    if not affected_vars or affected_vars[0] not in shift_details:
        return None

    num_regimes = int(non_stationarity_info.get('num_shifts', 0)) + 1
    labels = np.zeros(T, dtype=int)
    for shift in shift_details[affected_vars[0]]:
        regime = min(int(shift['regime']), num_regimes - 1)
        labels[shift['start']:shift['end']] = regime

    one_hot = np.zeros((T, num_regimes))
    one_hot[np.arange(T), labels] = 1.0
    return one_hot
