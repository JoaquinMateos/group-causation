import numpy as np
import pytest
import torch

from group_causation.aggregation_consistency import AggregationScore
from group_causation.group_causal_discovery.proposal_non_stationarity import (
    IVAE_GroupPCMCI_Proposal,
)

FAST_IVAE_PARAMS = dict(
    batch_size=64, max_epoch=10, seed=42,
    n_layers=1, hidden_dim=32, lr=1e-3,
    early_stopping_patience=3, anneal=False,
)


@pytest.fixture
def data() -> np.ndarray:
    return np.random.randn(100, 6).astype(np.float64)


@pytest.fixture
def groups() -> list[set[int]]:
    return [{0, 1}, {2, 3}, {4, 5}]


@pytest.fixture
def pcmci_params() -> dict:
    return {"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1}


@pytest.fixture
def numpy_u(data: np.ndarray) -> np.ndarray:
    return np.random.randint(0, 2, size=(data.shape[0], 3)).astype(np.float64)


def _make_proposal(data, groups, numpy_u, **overrides):
    """Helper to build a proposal with fast defaults."""
    params = dict(FAST_IVAE_PARAMS)
    params.update(overrides.pop("ivae_params", {}))
    return IVAE_GroupPCMCI_Proposal(
        data, groups, u=numpy_u, ivae_params=params,
        pcmci_params=overrides.get("pcmci_params", {"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1}),
        **{k: v for k, v in overrides.items() if k not in ("ivae_params", "pcmci_params")},
    )


class TestProposalInit:
    def test_default_initialization(self, data, groups, numpy_u, pcmci_params):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u, pcmci_params=pcmci_params,
        )
        assert inst.tau_max == 1
        assert inst.pc_alpha == 0.05

    def test_init_with_time_index_u(self, data, groups, pcmci_params):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u="time_index", num_chunks_of_time_index=5, pcmci_params=pcmci_params,
        )
        assert inst.u is not None
        assert inst.u.shape[1] == 5

    def test_init_raises_on_bad_ci_test(self, data, groups, numpy_u):
        with pytest.raises(ValueError, match="Unsupported independence test"):
            IVAE_GroupPCMCI_Proposal(
                data, groups, u=numpy_u,
                conditional_independence_test="nonexistent",
                pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
            )

    def test_raises_on_missing_tau_max(self, data, groups, numpy_u):
        with pytest.raises(KeyError):
            IVAE_GroupPCMCI_Proposal(
                data, groups, u=numpy_u,
                pcmci_params={"pc_alpha": 0.05, "max_conds_dim": 1},
            )

    def test_explicit_latent_dims_override_fraction(self, data, groups, numpy_u):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u, latent_dims=[1, 1, 1],
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        assert inst._fallback_dims == [1, 1, 1]

    def test_latent_dims_wrong_length_raises(self, data, groups, numpy_u):
        with pytest.raises(ValueError, match="Expected 3 latent dimensions"):
            IVAE_GroupPCMCI_Proposal(
                data, groups, u=numpy_u, latent_dims=[1, 1],
                pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
            )

    def test_latent_dims_out_of_range_raises(self, data, groups, numpy_u):
        with pytest.raises(ValueError, match="must be in"):
            IVAE_GroupPCMCI_Proposal(
                data, groups, u=numpy_u, latent_dims=[1, 1, 5],
                pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
            )

    def test_non_stationarity_shift_fallbacks_to_time_index(self, data, groups):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups,
            u="non_stationarity_shift", num_chunks_of_time_index=3,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
            non_stationarity_info={"type": "regime_shifts", "affected_vars": []},
        )
        assert inst is not None


class TestProposalExtractParents:
    @pytest.mark.slow
    def test_extract_parents_returns_dict(self, data, groups, numpy_u):
        inst = _make_proposal(data, groups, numpy_u,
                              pcmci_params={"tau_max": 1, "pc_alpha": 0.99, "max_conds_dim": 1})
        parents = inst.extract_parents()
        assert isinstance(parents, dict)
        for j in range(len(groups)):
            assert j in parents

    @pytest.mark.slow
    def test_extract_parents_with_adag_disabled(self, data, groups, numpy_u):
        inst = _make_proposal(data, groups, numpy_u,
                              apply_adag_optimization=False,
                              pcmci_params={"tau_max": 1, "pc_alpha": 0.99, "max_conds_dim": 1})
        parents = inst.extract_parents()
        assert isinstance(parents, dict)

    @pytest.mark.slow
    def test_extract_parents_with_pc_alpha_one(self, data, groups, numpy_u):
        inst = _make_proposal(data, groups, numpy_u,
                              pcmci_params={"tau_max": 1, "pc_alpha": 1.0, "max_conds_dim": 1})
        parents = inst.extract_parents()
        assert isinstance(parents, dict)
        for node in range(len(groups)):
            assert node in parents

    @pytest.mark.slow
    def test_extract_parents_with_hsic_and_regimes_falls_back(self, data, groups, numpy_u):
        inst = _make_proposal(data, groups, numpy_u,
                              conditional_independence_test="hsic",
                              pcmci_params={"tau_max": 1, "pc_alpha": 0.99, "max_conds_dim": 1})
        parents = inst.extract_parents()
        assert isinstance(parents, dict)


class TestProposalInternal:
    def test_get_device_returns_cpu(self, numpy_u):
        data = np.random.randn(50, 3).astype(np.float64)
        groups = [{0}, {1}, {2}]
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        dev = inst._get_device()
        assert str(dev) == "cpu"

    def test_raw_group_data_tensors(self, data, groups, numpy_u):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        for t in inst._raw_group_data:
            assert isinstance(t, torch.Tensor)
            assert t.shape[0] == data.shape[0]

    def test_fallback_dims_positive(self, data, groups, numpy_u):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        for d in inst._fallback_dims:
            assert d >= 1

    def test_get_effect_val_matrix_before_run_raises(self, data, groups, numpy_u):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        with pytest.raises(ValueError, match="Effect value matrix is not available"):
            inst.get_effect_val_matrix()

    @pytest.mark.slow
    def test_get_effect_val_matrix_after_run(self, data, groups, numpy_u):
        inst = _make_proposal(data, groups, numpy_u,
                              pcmci_params={"tau_max": 1, "pc_alpha": 0.5, "max_conds_dim": 1})
        inst.extract_parents()
        matrix = inst.get_effect_val_matrix()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (len(groups), len(groups), 2)

    def test_get_scores_before_run_raises(self, data, groups, numpy_u):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        with pytest.raises(ValueError, match="Aggregation scores are not available"):
            inst.get_scores()

    @pytest.mark.slow
    def test_get_scores_after_run(self, data, groups, numpy_u):
        inst = _make_proposal(data, groups, numpy_u,
                              pcmci_params={"tau_max": 1, "pc_alpha": 0.5, "max_conds_dim": 1})
        inst.extract_parents()
        score = inst.get_scores()
        assert isinstance(score, AggregationScore)
        assert 0.0 <= score.c_ind <= 1.0
        assert 0.0 <= score.c_dep <= 1.0
        assert score.ac == pytest.approx((score.c_ind + score.c_dep) / 2.0)


class TestProposalComputeCInd:
    def test_compute_c_ind_empty_returns_one(self, data, groups, numpy_u):
        inst = IVAE_GroupPCMCI_Proposal(
            data, groups, u=numpy_u,
            pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1},
        )
        score = inst._compute_c_ind([])
        assert score == 1.0


class TestProposalEdgeCases:
    @pytest.mark.slow
    def test_single_group(self, numpy_u):
        data = np.random.randn(50, 3).astype(np.float64)
        u = np.random.randint(0, 2, size=(data.shape[0], 3)).astype(np.float64)
        groups = [{0, 1, 2}]
        inst = _make_proposal(data, groups, u,
                              pcmci_params={"tau_max": 1, "pc_alpha": 0.05, "max_conds_dim": 1})
        parents = inst.extract_parents()
        assert 0 in parents

    @pytest.mark.slow
    def test_verbose_mode(self, data, groups, numpy_u, caplog):
        caplog.set_level("INFO")
        inst = _make_proposal(data, groups, numpy_u, verbose=1,
                              pcmci_params={"tau_max": 1, "pc_alpha": 0.99, "max_conds_dim": 1})
        inst.extract_parents()
        assert len(caplog.records) > 0
