# AGENTS.md

## Setup

```sh
uv sync
```

Python 3.14+ (`requires-python = ">=3.14"` in pyproject.toml). uv manages the Python version and virtualenv automatically.

## Run tests

```sh
uv run pytest
```

No lint, typecheck, or formatter is configured for this repo.

Slow tests (PyTorch model training) are marked with `@pytest.mark.slow`.

## Test quirks

`tests/conftest.py` has an `autouse` fixture that **mocks `torch.cuda.is_available` and `torch.backends.mps.is_available`** to return `False`. All tests run CPU-only with a seeded RNG (`np.random.seed(42)`, `torch.manual_seed(42)`).

## Structure

- `group_causation/shared_mixins.py` — `StandardizationMixin` + `MemoryMonitorMixin` (shared by all ABCs)
- `group_causation/causal_discovery_base.py` — `CausalDiscovery` ABC (leaf-level)
- `group_causation/group_causal_discovery/` — group-level algorithms (subclasses of `GroupCausalDiscovery` which extends `CausalDiscovery`)
- `group_causation/benchmark/` — benchmark harness; entry point is `BenchmarkGroupCausalDiscovery`
- `group_causation/data_management/` — synthetic dataset generation (`create_toy_datasets.py`, `time_series_generator.py`)
- `group_causation/groups_extraction/` — extracting variable groups from causal graphs
- `group_causation/dimensionality_reduction/` — PCA, adag wrappers, iVAE
- `group_causation/independence_tests/` — conditional independence tests (parcorr, HSIC, max_corr, shift-based variants)
- `group_causation/utils.py` — parameter generators for benchmark sweeps
- `examples/` — runnable benchmark scripts (e.g. `benchmark_increasing_nshifts.py`)

## Environment gotcha

GPU benchmarks need `os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'` set before torch imports. If MPS is available, the code forces `multiprocessing.set_start_method('spawn')` to avoid shared-memory issues with PyTorch.

## Docs

Sphinx docs live in `sphinx-docs/`. Build with `make html` inside that directory.
