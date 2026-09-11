import numpy as np
import pytest
import torch

from group_causation.aggregation_consistency import (
    AggregationConsistencyEvaluator,
    AggregationScore,
    InsufficientDataError,
    adjacency_statements,
    consistency_ratio,
)
from group_causation.dimensionality_reduction.adag_wrapper import AdagWrapper


class StubTester:
    """Statement tester that returns pre-scripted p-values."""

    def __init__(self, p_values):
        self._p_values = list(p_values)

    def __call__(self, statement):
        value = self._p_values.pop(0)
        if isinstance(value, Exception):
            raise value
        return 0.0, value


class AlwaysIndependent:
    @classmethod
    def test(cls, X, Y):
        return 0.0, 0.9

    @classmethod
    def conditional_test(cls, X, Y, Z):
        return 0.0, 0.9


class AlwaysDependent:
    @classmethod
    def test(cls, X, Y):
        return 1.0, 0.001

    @classmethod
    def conditional_test(cls, X, Y, Z):
        return 1.0, 0.001


def _make_adag(ci_test_class, max_lag=2):
    return AdagWrapper(
        ci_test_class=ci_test_class,
        groups=[[0], [1]],
        max_lag=max_lag,
        discovery_class=None,
        aggregator=None,
    )


class TestConsistencyRatio:
    def test_empty_defaults_to_one(self):
        assert consistency_ratio([]) == 1.0

    def test_all_consistent(self):
        assert consistency_ratio([True, True]) == 1.0

    def test_partially_consistent(self):
        assert consistency_ratio([True, False, False, False]) == pytest.approx(0.25)


class TestAdjacencyStatements:
    def test_builds_conditioning_without_the_tested_parent(self):
        statements = adjacency_statements({1: [(0, -1), (2, 0)]})
        assert statements == [
            (0, -1, 1, 0, [(2, 0)]),
            (2, 0, 1, 0, [(0, -1)]),
        ]

    def test_empty_graph_yields_no_statements(self):
        assert adjacency_statements({0: [], 1: []}) == []


class TestAggregationConsistencyEvaluator:
    def test_c_ind_counts_independent_statements(self):
        evaluator = AggregationConsistencyEvaluator(StubTester([0.5, 0.01]), alpha=0.05)
        assert evaluator.c_ind(['a', 'b']) == pytest.approx(0.5)

    def test_c_dep_counts_dependent_statements(self):
        evaluator = AggregationConsistencyEvaluator(StubTester([0.5, 0.01]), alpha=0.05)
        assert evaluator.c_dep(['a', 'b']) == pytest.approx(0.5)

    def test_untestable_statements_are_not_penalized(self):
        independent_evaluator = AggregationConsistencyEvaluator(StubTester([InsufficientDataError('too few')]), alpha=0.05)
        dependent_evaluator = AggregationConsistencyEvaluator(StubTester([InsufficientDataError('too few')]), alpha=0.05)
        assert independent_evaluator.c_ind(['a']) == 1.0
        assert dependent_evaluator.c_dep(['a']) == 1.0

    def test_evaluate_returns_score_with_averaged_ac(self):
        evaluator = AggregationConsistencyEvaluator(StubTester([0.5, 0.01, 0.001]), alpha=0.05)
        score = evaluator.evaluate(independencies=['a', 'b'], dependencies=['c'])
        assert isinstance(score, AggregationScore)
        assert score.c_ind == pytest.approx(0.5)
        assert score.c_dep == pytest.approx(1.0)
        assert score.ac == pytest.approx(0.75)


class TestAdagWrapperConsistency:
    @pytest.fixture
    def raw_data(self):
        return [torch.randn(50, 3), torch.randn(50, 2)]

    def test_requires_raw_data(self):
        wrapper = _make_adag(AlwaysIndependent)
        with pytest.raises(RuntimeError, match='Raw group data is missing'):
            wrapper._compute_c_ind({1: [(0, -1)]})

    def test_c_ind_is_one_when_vectors_are_independent(self, raw_data):
        wrapper = _make_adag(AlwaysIndependent)
        wrapper._raw_group_data = raw_data
        assert wrapper._compute_c_ind({1: [(0, -1)]}) == 1.0

    def test_c_ind_is_zero_when_vectors_are_dependent(self, raw_data):
        wrapper = _make_adag(AlwaysDependent)
        wrapper._raw_group_data = raw_data
        assert wrapper._compute_c_ind({1: [(0, -1)]}) == 0.0

    def test_c_dep_is_one_when_vectors_are_dependent(self, raw_data):
        wrapper = _make_adag(AlwaysDependent)
        wrapper._raw_group_data = raw_data
        assert wrapper._compute_c_dep({1: [(0, -1)]}) == 1.0

    def test_c_dep_is_zero_when_vectors_are_independent(self, raw_data):
        wrapper = _make_adag(AlwaysIndependent)
        wrapper._raw_group_data = raw_data
        assert wrapper._compute_c_dep({1: [(0, -1)]}) == 0.0

    def test_untestable_adjacency_is_treated_as_consistent(self):
        wrapper = _make_adag(AlwaysDependent)
        wrapper._raw_group_data = [torch.randn(6, 3), torch.randn(6, 2)]
        assert wrapper._compute_c_dep({1: [(0, -1)]}) == 1.0
