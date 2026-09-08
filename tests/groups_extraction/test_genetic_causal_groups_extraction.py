from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from group_causation.groups_extraction.genetic_causal_groups_extraction import (
    GeneticCausalGroupsExtractor,
    crossover_partitions,
    _run_genetic_algorithm,
)


class TestCrossoverPartitions:
    def test_crossover_returns_two_partitions(self):
        p1 = [{0, 1}, {2, 3}]
        p2 = [{0}, {1, 2, 3}]
        r1, r2 = crossover_partitions(p1, p2)
        assert len(r1) >= 1
        assert len(r2) >= 1

    def test_crossover_covers_universe(self):
        p1 = [{0, 1}, {2, 3}]
        p2 = [{0, 2}, {1, 3}]
        r1, r2 = crossover_partitions(p1, p2)
        universe = {0, 1, 2, 3}
        for result in (r1, r2):
            union = set().union(*(s for s in result))
            assert union == universe

    def test_crossover_disjoint_groups(self):
        p1 = [{0, 1}, {2}, {3, 4}]
        p2 = [{0, 3}, {1, 4}, {2}]
        r1, r2 = crossover_partitions(p1, p2)
        for result in (r1, r2):
            seen: set[int] = set()
            for g in result:
                assert seen.isdisjoint(g)
                seen.update(g)

    def test_crossover_single_group_partition(self):
        p1 = [{0, 1, 2}]
        p2 = [{0, 1, 2}]
        r1, r2 = crossover_partitions(p1, p2)
        for result in (r1, r2):
            union = set().union(*(s for s in result))
            assert union == {0, 1, 2}


class TestGeneticCausalGroupsExtractor:
    @pytest.fixture
    def data(self) -> np.ndarray:
        return np.random.randn(100, 6).astype(np.float64)

    def test_extract_groups_returns_list_of_sets(self, data: np.ndarray):
        extractor = GeneticCausalGroupsExtractor(
            data, scores=["explainability_score"], scores_weights=[1.0]
        )
        groups = extractor.extract_groups()
        assert isinstance(groups, list)
        for g in groups:
            assert isinstance(g, set)

    def test_all_nodes_covered(self, data: np.ndarray):
        extractor = GeneticCausalGroupsExtractor(
            data, scores=["explainability_score"], scores_weights=[1.0]
        )
        groups = extractor.extract_groups()
        all_nodes = set()
        for g in groups:
            all_nodes.update(g)
        assert all_nodes == set(range(data.shape[1]))

    def test_disjoint_groups(self, data: np.ndarray):
        extractor = GeneticCausalGroupsExtractor(
            data, scores=["explainability_score"], scores_weights=[1.0]
        )
        groups = extractor.extract_groups()
        seen: set[int] = set()
        for g in groups:
            assert seen.isdisjoint(g)
            seen.update(g)

    def test_single_variable(self):
        data = np.random.randn(50, 1).astype(np.float64)
        extractor = GeneticCausalGroupsExtractor(
            data, scores=["explainability_score"], scores_weights=[1.0]
        )
        groups = extractor.extract_groups()
        assert groups == [{0}]

    def test_multi_objective(self, data: np.ndarray):
        extractor = GeneticCausalGroupsExtractor(
            data,
            scores=["average_variance_explained", "explainability_score"],
            scores_weights=[1.0, -1.0],
        )
        groups = extractor.extract_groups()
        assert len(groups) > 0

    def test_run_genetic_algorithm_deterministic_seed(self):
        import random as rnd
        rnd.seed(0)
        np.random.seed(0)
        a = _run_genetic_algorithm(5, lambda x: [1.0], scores_weights=[1.0])
        rnd.seed(0)
        np.random.seed(0)
        b = _run_genetic_algorithm(5, lambda x: [1.0], scores_weights=[1.0])
        assert a == b

    def test_run_genetic_algorithm_empty_groups_removed(self):
        result = _run_genetic_algorithm(3, lambda x: [1.0], scores_weights=[1.0])
        for g in result:
            assert len(g) > 0

    @pytest.mark.parametrize("n_vars", [2, 5, 10])
    def test_run_genetic_algorithm_different_sizes(self, n_vars: int):
        result = _run_genetic_algorithm(n_vars, lambda x: [float(n_vars)], scores_weights=[1.0])
        all_nodes = set().union(*result)
        assert all_nodes == set(range(n_vars))


class TestRunGeneticAlgorithmEdgeCases:
    def test_scores_weights_mutable_default_not_shared(self):
        score_fn = lambda x: [1.0]
        r1 = _run_genetic_algorithm(4, score_fn)
        r2 = _run_genetic_algorithm(4, score_fn)
        assert r1 is not r2
