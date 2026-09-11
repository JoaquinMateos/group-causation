import numpy as np
import pytest
import torch

from group_causation.dimensionality_reduction.iVAE.nets import iVAE
from group_causation.dimensionality_reduction.iVAE.wrappers import IVAEWrapper


def _make_ivae(consistency_weight: float = 0.0, seed: int = 0) -> iVAE:
    torch.manual_seed(seed)
    return iVAE(
        latent_dim=2, data_dim=4, aux_dim=3,
        n_layers=1, hidden_dim=8, consistency_weight=consistency_weight,
    )


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(64, 4), torch.randn(64, 3)


class TestIVAEConsistencyLoss:
    def test_consistency_weight_reduces_objective(self, batch):
        x, u = batch
        elbo_plain, _ = _make_ivae(consistency_weight=0.0).elbo(x, u)
        elbo_regularized, _ = _make_ivae(consistency_weight=0.5).elbo(x, u)
        assert elbo_regularized.detach().item() < elbo_plain.detach().item()

    def test_gradients_flow_through_consistency_term(self, batch):
        x, u = batch
        model = _make_ivae(consistency_weight=0.5)
        elbo, _ = model.elbo(x, u)
        (-elbo).backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert grads
        assert any(float(grad.abs().sum()) > 0 for grad in grads)

    def test_zero_weight_keeps_plain_objective(self, batch):
        x, u = batch
        model = _make_ivae(consistency_weight=0.0)
        assert model.consistency_weight == 0.0
        elbo, _ = model.elbo(x, u)
        assert torch.isfinite(elbo)


class TestIVAEWrapperConsistency:
    def test_wrapper_trains_with_consistency_weight(self):
        wrapper = IVAEWrapper(
            latent_dim=2, batch_size=16, max_epoch=2, n_layers=1, hidden_dim=8,
            lr=1e-2, seed=0, consistency_weight=0.1, anneal=False,
        )
        latents = wrapper.fit_transform(np.random.randn(64, 4), np.random.randn(64, 3))
        assert latents.shape == (64, 2)
        assert torch.isfinite(latents).all()

    def test_default_wrapper_has_no_consistency_penalty(self):
        wrapper = IVAEWrapper(latent_dim=2, batch_size=16, max_epoch=1, n_layers=1, hidden_dim=8, seed=0)
        assert wrapper.consistency_weight == 0.0
