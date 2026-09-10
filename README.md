# Group Causation

A Python library for causal discovery on time series and **groups of time series**. It provides algorithms for identifying causal relationships, tools for non-stationary data with regime shifts, and a benchmarking framework for systematic evaluation.

![Architecture](https://github.com/user-attachments/assets/25aa8679-4185-4d1a-808e-5527c80a301d)

## Installation

Requires Python 3.14+. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
git clone https://github.com/JoaquinMateosBarroso/group-causation
cd group-causation
uv sync
```

## Quick start

### Generate a synthetic dataset

```python
from group_causation.data_management.create_toy_datasets import CausalDataset

ds = CausalDataset()
time_series, parents, groups, node_parents, ns_info = ds.generate_group_toy_data(
    name='example', T=1000, N_vars=20, N_groups=4,
    max_lag=3, min_lag=1,
    dependency_funcs=['linear'],
    dependency_coeffs=[-0.5, 0.5],
    auto_coeffs=[0.6],
    noise_dists=['gaussian'],
    noise_sigmas=[0.2],
)
```

### Run a group causal discovery algorithm

```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from group_causation.group_causal_discovery import GroupPCMCICausalDiscovery

algorithm = GroupPCMCICausalDiscovery(
    time_series, groups,
    tau_max=3, pc_alpha=0.05, max_conds_dim=2,
    u='time_index', num_chunks_of_time_index=1,
)
parents = algorithm.extract_parents()
```

### Measure time and memory

```python
parents, elapsed, memory_mb = algorithm.extract_parents_time_and_memory()
print(f"Took {elapsed:.2f}s, {memory_mb:.1f} MiB peak RSS")
```

## Algorithms

### Node-level (single time series)

| Method | Class |
|---|---|
| PCMCI | `PCMCIWrapper` |
| PC-Stable | `PCMCIWrapper` (via `cond_ind_test`) |
| Granger Causality | `GrangerWrapper` |
| VAR-LiNGAM | `VARLINGAMWrapper` |
| DYNOTEARS | `DynotearsWrapper` |

### Group-level (groups of time series)

| Method | Class | Description |
|---|---|---|
| Micro-level | `MicroLevelGroupCausalDiscovery` | Node-level CD, then aggregate to groups |
| PCA + CD | `DimensionReductionGroupCausalDiscovery` | PCA per group, then node-level CD |
| Adag Embedding | `HybridGroupCausalDiscovery` | PCA + Adag optimization for latent dims |
| Group-PCMCI | `GroupPCMCICausalDiscovery` | Group-level PCMCI with localized tests |
| GroupRESIT | `GroupRESITTimeSeriesCausalDiscovery` | Neural network-based residual independence |
| gCDMI | `gCDMICausalDiscovery` | Deep mutual information with knockoffs |
| iVAE + PCMCI | `IVAE_GroupPCMCI_Proposal` | Identifiable VAE for non-stationary data |

### Conditional independence tests

| Test | Class | Use case |
|---|---|---|
| Max-Corr | `MaxCorr_Test` | Non-linear, fast, uses CCT aggregation |
| HSIC | `HSIC_Test` | Kernel-based, any dependency |
| Localized ParCorr | via tigramite | Per-chunk Pearson correlation |
| Shift-based | via tigramite | Uses regime shift information |

### Group extraction

| Method | Class |
|---|---|
| Random search | `RandomCausalGroupsExtractor` |
| Genetic algorithm | `GeneticCausalGroupsExtractor` |
| Exhaustive search | `ExhaustiveCausalGroupsExtractor` |

## Benchmarking

The benchmarking framework automates comparison across algorithms, datasets, and hyperparameter sweeps.

```python
from group_causation.benchmark import BenchmarkGroupCausalDiscovery

with BenchmarkGroupCausalDiscovery(info_file='info.log', debug_file='debug.log') as bm:
    results = bm.benchmark_causal_discovery(
        algorithms=algorithms,
        parameters_iterator=parameters_iterator,
        datasets_folder='./toy_data',
        generate_toy_data=True,
        results_folder='./results',
        n_executions=5,
        max_parallel_executions=3,
    )
    bm.plot_moving_results('./results', x_axis='num_shifts',
                           scores=['f1', 'precision', 'recall', 'shd'])
```

### Built-in parameter sweeps

```python
from group_causation.utils import (
    changing_N_groups,
    changing_N_variables,
    changing_non_stationarity_params,
    changing_latent_confounding_fraction,
    changing_alg_params,
)
```

Each generator yields `(algorithm_params, data_options)` tuples that the benchmark iterates over.

## Examples

See [`examples/`](examples/) for runnable scripts:

| Script | What it does |
|---|---|
| `benchmark_increasing_nshifts.py` | Sweep over increasing non-stationarity (regime shifts) |
| `benchmark_increasing_naffected_vars.py` | Sweep over number of affected variables |
| `sensitivity_analysis.py` | Hyperparameter sensitivity for iVAE + PCMCI |

Run an example:

```sh
uv run python examples/benchmark_increasing_nshifts.py
```

## Architecture

```
group_causation/
├── shared_mixins.py           # StandardizationMixin + MemoryMonitorMixin
├── causal_discovery_base.py   # CausalDiscovery ABC
├── group_causal_discovery/    # Group-level algorithms
├── micro_causal_discovery/    # Node-level algorithms (PCMCI, DYNOTEARS, Granger)
├── independence_tests/        # HSIC, Max-Corr, conditional independence
├── dimensionality_reduction/  # PCA, Adag wrapper, iVAE
├── groups_extraction/         # Random, genetic, exhaustive group search
├── data_management/           # Synthetic dataset generation
├── benchmark/                 # Benchmark harness
└── utils.py                   # Parameter generators, metrics, graph utilities
```

## License

MIT

## Contact

Open an issue or contact [jmateosbarroso@gmail.com](mailto:jmateosbarroso@gmail.com).
