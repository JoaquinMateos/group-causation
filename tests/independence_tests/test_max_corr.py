import numpy as np
import pytest
import torch

from group_causation.independence_tests.max_corr import MaxCorr_Test


@pytest.fixture
def X() -> torch.Tensor:
    return torch.randn(200, 1)


@pytest.fixture
def Y() -> torch.Tensor:
    return torch.randn(200, 1)


@pytest.fixture
def dependent_xy() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(200, 1)
    y = x * 0.8 + torch.randn(200, 1) * 0.6
    return x, y


class TestMaxCorrSingleTest:
    def test_single_test_independent(self, X: torch.Tensor, Y: torch.Tensor):
        stat, pval = MaxCorr_Test._single_test(X, Y)
        assert isinstance(stat, float)
        assert isinstance(pval, float)
        assert 0.0 <= pval <= 1.0

    def test_single_test_dependent(self, dependent_xy: tuple[torch.Tensor, torch.Tensor]):
        x, y = dependent_xy
        stat, pval = MaxCorr_Test._single_test(x, y)
        assert pval < 0.05

    def test_single_test_too_few_samples(self):
        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        stat, pval = MaxCorr_Test._single_test(x, y)
        assert pval == 1.0
        assert stat == 0.0

    def test_multivariate_input(self):
        x = torch.randn(100, 3)
        y = torch.randn(100, 2)
        stat, pval = MaxCorr_Test._single_test(x, y)
        assert 0.0 <= pval <= 1.0
        assert stat >= 0.0


class TestMaxCorrTest:
    def test_test_small_data(self, X: torch.Tensor, Y: torch.Tensor):
        stat, pval = MaxCorr_Test.test(X, Y, max_samples=500)
        assert 0.0 <= pval <= 1.0

    def test_test_large_data_random_subsample(self):
        x = torch.randn(2000, 1)
        y = torch.randn(2000, 1)
        stat, pval = MaxCorr_Test.test(x, y, max_samples=100, n_ensembles=3)
        assert 0.0 <= pval <= 1.0

    def test_test_sequential_chunks(self):
        x = torch.randn(2000, 1)
        y = torch.randn(2000, 1)
        stat, pval = MaxCorr_Test.test(x, y, max_samples=100, sequential_chunks=True)
        assert 0.0 <= pval <= 1.0

    def test_test_empty_regimes(self):
        stat, pval = MaxCorr_Test.test_regimes([], [])
        assert stat == 0.0
        assert pval == 1.0

    def test_regime_test_single(self):
        x_reg = [torch.randn(100, 1)]
        y_reg = [torch.randn(100, 1)]
        stat, pval = MaxCorr_Test.test_regimes(x_reg, y_reg)
        assert 0.0 <= pval <= 1.0

    def test_regime_test_multi(self):
        x_reg = [torch.randn(50, 1), torch.randn(60, 1)]
        y_reg = [torch.randn(50, 1), torch.randn(60, 1)]
        stat, pval = MaxCorr_Test.test_regimes(x_reg, y_reg)
        assert 0.0 <= pval <= 1.0


class TestMaxCorrConditional:
    def test_single_conditional_test(self, X: torch.Tensor, Y: torch.Tensor):
        Z = torch.randn(200, 2)
        stat, pval = MaxCorr_Test._single_conditional_test(X, Y, Z, ridge_lambda=0.2)
        assert 0.0 <= pval <= 1.0

    def test_conditional_test_small(self, X: torch.Tensor, Y: torch.Tensor):
        Z = torch.randn(200, 2)
        stat, pval = MaxCorr_Test.conditional_test(X, Y, Z, max_samples=500)
        assert 0.0 <= pval <= 1.0

    def test_conditional_regime_test(self):
        x_reg = [torch.randn(50, 1), torch.randn(50, 1)]
        y_reg = [torch.randn(50, 1), torch.randn(50, 1)]
        z_reg = [torch.randn(50, 2), torch.randn(50, 2)]
        stat, pval = MaxCorr_Test.conditional_test_regimes(x_reg, y_reg, z_reg)
        assert 0.0 <= pval <= 1.0

    def test_conditional_regime_test_empty(self):
        stat, pval = MaxCorr_Test.conditional_test_regimes([], [], [])
        assert stat == 0.0
        assert pval == 1.0


class TestMaxCorrAggregation:
    def test_cct_aggregation_uniform_weights(self):
        stats = [1.0, 2.0, 0.5]
        p_vals = [0.1, 0.2, 0.3]
        weights = [1 / 3, 1 / 3, 1 / 3]
        avg_stat, global_p = MaxCorr_Test._aggregate_cct(stats, p_vals, weights)
        assert avg_stat == pytest.approx(sum(w * s for w, s in zip(weights, stats)))
        assert 0.0 <= global_p <= 1.0


class TestMaxCorrCapabilities:
    def test_regime_testing_is_supported(self):
        assert MaxCorr_Test.supports_regime_testing is True


class TestMaxCorrEdgeCases:
    def test_1d_input_reshaped(self):
        x = torch.randn(100)
        y = torch.randn(100)
        stat, pval = MaxCorr_Test.test(x, y, max_samples=500)
        assert 0.0 <= pval <= 1.0

    def test_constant_input(self):
        x = torch.ones(100, 1)
        y = torch.randn(100, 1)
        stat, pval = MaxCorr_Test._single_test(x, y)
        assert pval >= 0.0

    def test_multiple_regimes_some_too_small(self):
        x_reg = [torch.randn(50, 1), torch.randn(3, 1)]
        y_reg = [torch.randn(50, 1), torch.randn(3, 1)]
        stat, pval = MaxCorr_Test.test_regimes(x_reg, y_reg)
        assert 0.0 <= pval <= 1.0
