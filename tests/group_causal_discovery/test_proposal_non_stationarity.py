import numpy as np
import pytest
import torch

from group_causation.group_causal_discovery.proposal_non_stationarity import (
    IVAE_GroupPCMCI_Proposal,
)


@pytest.fixture
def data() -> np.ndarray:
    return np.random.randn(200, 6).astype(np.float64)


@pytest.fixture
def groups() -> list[set[int]]:
    return [{0, 1}, {2, 3}, {4, 5}]


@pytest.fixture
def pcmci_params() -> dict:
    return {"tau_max": 2, "pc_alpha": 0.05, "max_conds_dim": 2}


@pytest.fixture
def numpy_u(data: np.ndarray) -> np.ndarray:
    return np.random.randint(0, 2, size=(data.shape[0], 3)).astype(np.float64)


class TestProposalInit:
    def test_default_initialization(self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray, pcmci_params: dict):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u, pcmci_params=pcmci_params,
        )
        assert inst.tau_max == 2
        assert inst.pc_alpha == 0.05

    def test_init_with_time_index_u(self, data: np.ndarray, groups: list[set[int]], pcmci_params: dict):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u="time_index", num_chunks_of_time_index=5, pcmci_params=pcmci_params,
        )
        assert inst.u is not None
        assert inst.u.shape[1] == 5

    def test_init_raises_on_bad_ci_test(self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray):
        with pytest.raises(ValueError, match="Unsupported independence test"):
            IVAE_GroupPCMCI_Proposal(
                data, groups, u=numpy_u,
                conditional_independence_test="nonexistent",
                pcmci_params={"tau_max": 2, "pc_alpha": 0.05, "max_conds_dim": 2},
            )

    def test_raises_on_missing_tau_max(self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray):
        with pytest.raises(KeyError):
            IVAE_GroupPCMCI_Proposal(
                data, groups, u=numpy_u,
                pcmci_params={"pc_alpha": 0.05, "max_conds_dim": 2},
            )

    def test_non_stationarity_shift_fallbacks_to_time_index(
        self, data: np.ndarray, groups: list[set[int]]
    ):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups,
            u="non_stationarity_shift", num_chunks_of_time_index=3,
            pcmci_params={"tau_max": 2, "pc_alpha": 0.05, "max_conds_dim": 2},
            non_stationarity_info={"type": "regime_shifts", "affected_vars": []},
        )
        assert inst is not None


class TestProposalExtractParents:
    @pytest.mark.slow
    def test_extract_parents_returns_dict(
        self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray
    ):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.99, "max_conds_dim": 1},
        )
        parents = inst.extract_parents()
        assert isinstance(parents, dict)
        for j in range(len(groups)):
            assert j in parents

    @pytest.mark.slow
    def test_extract_parents_with_adag_disabled(
        self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray
    ):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            apply_adag_optimization=False,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.99, "max_conds_dim": 1},
        )
        parents = inst.extract_parents()
        assert isinstance(parents, dict)

    @pytest.mark.slow
    def test_extract_parents_with_pc_alpha_one(
        self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray
    ):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 1.0, "max_conds_dim": 1},
        )
        parents = inst.extract_parents()
        for node, p_list in parents.items():
            assert len(p_list) == 0


class TestProposalInternal:
    def test_get_device_returns_cpu(self, numpy_u: np.ndarray):
        data = np.random.randn(50, 3).astype(np.float64)
        groups = [{0}, {1}, {2}]
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        dev = inst._get_device()
        assert str(dev) == "cpu"

    def test_raw_group_data_tensors(self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        for t in inst._raw_group_data:
            assert isinstance(t, torch.Tensor)
            assert t.shape[0] == data.shape[0]

    def test_fallback_dims_positive(self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        for d in inst._fallback_dims:
            assert d >= 1

    def test_get_effect_val_matrix_before_run_raises(
        self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray
    ):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        with pytest.raises(ValueError, match="Effect value matrix is not available"):
            inst.get_effect_val_matrix()

    @pytest.mark.slow
    def test_get_effect_val_matrix_after_run(
        self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray
    ):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.5, "max_conds_dim": 1},
        )
        inst.extract_parents()
        matrix = inst.get_effect_val_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (len(groups), len(groups), 2)


class TestProposalComputeCInd:
    def test_compute_c_ind_empty_returns_one(
        self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray
    ):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        score = inst._compute_c_ind([])
        assert score == 1.0


class TestProposalEdgeCases:
    @pytest.mark.slow
    def test_single_group(self, numpy_u: np.ndarray):
        data = np.random.randn(100, 3).astype(np.float64)
        groups = [{0, 1, 2}]
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        parents = inst.extract_parents()
        assert 0 in parents

    @pytest.mark.slow
    def test_verbose_mode(self, data: np.ndarray, groups: list[set[int]], numpy_u: np.ndarray, caplog):
        caplog.set_level("INFO")
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u, verbose=1,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.99, "max_conds_dim": 1},
        )
        inst.extract_parents()
        assert len(caplog.records) > 0
