import torch

from group_causation.independence_tests.dhsic import dhsic, pairwise_latent_dhsic


class TestDHSIC:
    def test_independent_samples_score_low(self):
        torch.manual_seed(0)
        X = torch.randn(300, 1)
        Y = torch.randn(300, 1)
        assert float(dhsic(X, Y)) < 0.5

    def test_dependent_samples_score_high(self):
        torch.manual_seed(0)
        X = torch.randn(300, 1)
        Y = X ** 2 + 0.1 * torch.randn(300, 1)
        assert float(dhsic(X, Y)) > 5.0

    def test_multivariate_inputs_are_supported(self):
        torch.manual_seed(0)
        X = torch.randn(200, 3)
        Y = torch.randn(200, 2)
        assert dhsic(X, Y).shape == ()

    def test_statistic_keeps_computation_graph(self):
        torch.manual_seed(0)
        X = torch.randn(100, 2, requires_grad=True)
        Y = torch.randn(100, 2)
        dhsic(X, Y).backward()
        assert X.grad is not None
        assert torch.isfinite(X.grad).all()


class TestPairwiseLatentDHSIC:
    def test_single_latent_returns_zero(self):
        assert float(pairwise_latent_dhsic(torch.randn(100, 1))) == 0.0

    def test_dependent_latents_score_higher_than_independent(self):
        torch.manual_seed(0)
        independent = pairwise_latent_dhsic(torch.randn(300, 3))
        shared = torch.randn(300, 1)
        dependent = pairwise_latent_dhsic(shared @ torch.randn(1, 3) + 0.01 * torch.randn(300, 3))
        assert float(dependent) > float(independent)

    def test_gradients_flow_to_latents(self):
        torch.manual_seed(0)
        latents = torch.randn(100, 3, requires_grad=True)
        pairwise_latent_dhsic(latents).backward()
        assert latents.grad is not None
        assert torch.isfinite(latents.grad).all()
