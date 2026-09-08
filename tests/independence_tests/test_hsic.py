import numpy as np
import pytest
import torch

from group_causation.independence_tests.hsic import HSIC_Test


@pytest.fixture
def X() -> torch.Tensor:
    return torch.randn(150, 1)


@pytest.fixture
def Y() -> torch.Tensor:
    return torch.randn(150, 1)


@pytest.fixture
def dependent_xy() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(150, 1)
    y = x**2 + torch.randn(150, 1) * 0.3
    return x, y


class TestHSICKernel:
    def test_get_kernel_width(self, X: torch.Tensor):
        width = HSIC_Test.get_kernel_width(X)
        assert width > 0
        assert isinstance(width, float)

    def test_get_kernel_width_small_sample(self):
        x = torch.randn(5, 2)
        width = HSIC_Test.get_kernel_width(x)
        assert width > 0

    def test_get_gram_matrix(self, X: torch.Tensor):
        width = HSIC_Test.get_kernel_width(X)
        K, Kc = HSIC_Test.get_gram_matrix(X, width)
        assert K.shape == (X.shape[0], X.shape[0])
        assert Kc.shape == (X.shape[0], X.shape[0])


class TestHSICSingleTest:
    def test_single_test_independent(self, X: torch.Tensor, Y: torch.Tensor):
        stat, pval = HSIC_Test._single_test(X, Y)
        assert isinstance(stat, float)
        assert isinstance(pval, float)
        assert 0.0 <= pval <= 1.0

    def test_single_test_dependent(self, dependent_xy: tuple[torch.Tensor, torch.Tensor]):
        x, y = dependent_xy
        stat, pval = HSIC_Test._single_test(x, y)
        assert pval < 0.05

    def test_single_test_too_few_samples(self):
        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        stat, pval = HSIC_Test._single_test(x, y)
        assert pval == 1.0
        assert stat == 0.0


class TestHSICTest:
    def test_test_small(self, X: torch.Tensor, Y: torch.Tensor):
        stat, pval = HSIC_Test.test(X, Y, max_samples=500)
        assert 0.0 <= pval <= 1.0

    def test_test_large_random_subsample(self):
        x = torch.randn(2000, 1)
        y = torch.randn(2000, 1)
        stat, pval = HSIC_Test.test(x, y, max_samples=100, n_ensembles=3)
        assert 0.0 <= pval <= 1.0

    def test_test_sequential_chunks(self):
        x = torch.randn(2000, 1)
        y = torch.randn(2000, 1)
        stat, pval = HSIC_Test.test(x, y, max_samples=100, sequential_chunks=True)
        assert 0.0 <= pval <= 1.0

    def test_1d_input_reshaped(self):
        x = torch.randn(100)
        y = torch.randn(100)
        stat, pval = HSIC_Test.test(x, y, max_samples=500)
        assert 0.0 <= pval <= 1.0


class TestHSICConditional:
    def test_single_conditional_test(self, X: torch.Tensor, Y: torch.Tensor):
        Z = torch.randn(150, 2)
        stat, pval = HSIC_Test._single_conditional_test(X, Y, Z, epsilon=1e-3)
        assert 0.0 <= pval <= 1.0

    def test_conditional_test_small(self, X: torch.Tensor, Y: torch.Tensor):
        Z = torch.randn(150, 2)
        stat, pval = HSIC_Test.conditional_test(X, Y, Z, max_samples=500)
        assert 0.0 <= pval <= 1.0

    def test_conditional_test_too_few_samples(self):
        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        z = torch.randn(3, 1)
        stat, pval = HSIC_Test._single_conditional_test(x, y, z)
        assert pval == 1.0


class TestHSICEdgeCases:
    def test_constant_input(self):
        x = torch.ones(100, 1)
        y = torch.randn(100, 1)
        stat, pval = HSIC_Test._single_test(x, y)
        assert 0.0 <= pval <= 1.0

    def test_zero_variance(self):
        x = torch.zeros(100, 1)
        y = torch.ones(100, 1)
        stat, pval = HSIC_Test._single_test(x, y)
        assert 0.0 <= pval <= 1.0
