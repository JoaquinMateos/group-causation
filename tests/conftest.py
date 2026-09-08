from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _seed_all() -> Generator[None, None, None]:
    np.random.seed(42)
    torch.manual_seed(42)
    yield


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def small_data() -> np.ndarray:
    return np.random.randn(100, 5).astype(np.float64)


@pytest.fixture
def constant_feature_data() -> np.ndarray:
    data = np.random.randn(100, 4).astype(np.float64)
    const_col = np.ones((100, 1), dtype=np.float64) * 5.0
    return np.concatenate([data, const_col], axis=1)


@pytest.fixture
def zero_variance_data() -> np.ndarray:
    return np.ones((100, 5), dtype=np.float64) * 3.0


@pytest.fixture
def small_groups() -> list[set[int]]:
    return [{0, 1}, {2, 3}, {4}]


@pytest.fixture
def small_group_data(small_data: np.ndarray) -> np.ndarray:
    return small_data


@pytest.fixture
def tiny_window_graph() -> dict[int, list[tuple[int, int]]]:
    return {
        0: [(1, -1), (2, 0)],
        1: [(0, -2)],
        2: [],
    }


@pytest.fixture
def sample_parents_dict() -> dict[int, list[tuple[int, int]]]:
    return {
        0: [(1, -1), (2, 0)],
        1: [(0, -2), (3, -1)],
        2: [(1, -1)],
        3: [],
    }


# ---------------------------------------------------------------------------
# Shared mocks for heavy external dependencies
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _mock_torch_no_cuda() -> Generator[None, None, None]:
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        yield


@pytest.fixture
def mock_psutil() -> Generator[MagicMock, None, None]:
    with patch("psutil.Process") as mp:
        proc = MagicMock()
        proc.memory_info.return_value.rss = 100_000_000
        mp.return_value = proc
        yield mp


@pytest.fixture
def mock_tigramite_pcmci() -> Generator[MagicMock, None, None]:
    with patch("tigramite.pcmci.PCMCI") as mp:
        instance = MagicMock()
        instance.run_pcmciplus.return_value = {
            "graph": np.zeros((5, 5, 4)),
            "val_matrix": np.zeros((5, 5, 4)),
        }
        instance.return_parents_dict.return_value = {}
        mp.return_value = instance
        yield mp


@pytest.fixture
def caplog_verbose(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level("DEBUG")
    return caplog


# ---------------------------------------------------------------------------
# Benchmark / dataset helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_causal_dataset() -> dict[str, Any]:
    return {
        "time_series": np.random.randn(200, 6).astype(np.float64),
        "parents_dict": {
            0: [(1, -1), (2, 0)],
            1: [(0, -2)],
            2: [],
            3: [(4, -1)],
            4: [(5, -1)],
            5: [],
        },
        "groups": [[0, 1], [2], [3, 4, 5]],
        "node_parents_dict": {},
        "non_stationarity_info": {"applied": False},
    }


@pytest.fixture
def non_stationarity_info() -> dict[str, Any]:
    return {
        "applied": True,
        "type": "regime_shifts",
        "num_shifts": 2,
        "affected_vars": [0, 1],
        "shift_details": {
            0: [
                {"regime": 0, "start": 0, "end": 50, "mean_shift": 0.0, "std_mult": 1.0},
                {"regime": 1, "start": 50, "end": 100, "mean_shift": 3.0, "std_mult": 1.5},
                {"regime": 2, "start": 100, "end": 200, "mean_shift": -2.0, "std_mult": 0.8},
            ],
            1: [
                {"regime": 0, "start": 0, "end": 50, "mean_shift": 0.0, "std_mult": 1.0},
                {"regime": 1, "start": 50, "end": 100, "mean_shift": -1.0, "std_mult": 2.0},
                {"regime": 2, "start": 100, "end": 200, "mean_shift": 4.0, "std_mult": 1.2},
            ],
        },
    }
