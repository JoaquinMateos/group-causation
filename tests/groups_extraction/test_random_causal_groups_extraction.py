import numpy as np
import pytest

from group_causation.groups_extraction.random_causal_groups_extraction import (
    RandomCausalGroupsExtractor,
)


class TestRandomCausalGroupsExtractor:
    @pytest.fixture
    def data(self) -> np.ndarray:
        return np.random.randn(200, 10).astype(np.float64)

    def test_best_partition_is_actually_best(self, data: np.ndarray):
        extractor = RandomCausalGroupsExtractor(
            data, scores=["explainability_score"]
        )
        result = extractor.extract_groups()

        assert isinstance(result, list)
        assert all(isinstance(g, set) for g in result)
        assert len(result) > 0

    def test_all_nodes_are_accounted_for(self, data: np.ndarray):
        extractor = RandomCausalGroupsExtractor(
            data, scores=["explainability_score"]
        )
        groups = extractor.extract_groups()
        all_nodes = set()
        for g in groups:
            all_nodes.update(g)
        assert all_nodes == set(range(data.shape[1]))

    def test_groups_are_disjoint(self, data: np.ndarray):
        extractor = RandomCausalGroupsExtractor(
            data, scores=["explainability_score"]
        )
        groups = extractor.extract_groups()
        seen: set[int] = set()
        for g in groups:
            assert seen.isdisjoint(g), f"overlapping groups: {seen} ∩ {g}"
            seen.update(g)

    def test_no_empty_groups(self, data: np.ndarray):
        extractor = RandomCausalGroupsExtractor(
            data, scores=["explainability_score"]
        )
        groups = extractor.extract_groups()
        for g in groups:
            assert len(g) > 0, "empty group found"

    def test_single_variable_data(self):
        data = np.random.randn(50, 1).astype(np.float64)
        extractor = RandomCausalGroupsExtractor(
            data, scores=["explainability_score"]
        )
        groups = extractor.extract_groups()
        assert groups == [{0}]

    def test_two_variables(self):
        data = np.random.randn(50, 2).astype(np.float64)
        extractor = RandomCausalGroupsExtractor(
            data, scores=["explainability_score"]
        )
        groups = extractor.extract_groups()
        all_nodes = {n for g in groups for n in g}
        assert all_nodes == {0, 1}

    def test_extract_groups_with_average_variance_explained(self):
        data = np.random.randn(100, 6).astype(np.float64)
        extractor = RandomCausalGroupsExtractor(
            data, scores=["average_variance_explained"],
        )
        groups = extractor.extract_groups()
        assert len(groups) > 0

    def test_extract_groups_time_and_memory(self, data: np.ndarray):
        extractor = RandomCausalGroupsExtractor(
            data, scores=["explainability_score"]
        )
        groups, elapsed, mem = extractor.extract_groups_time_and_memory()
        assert isinstance(groups, list)
        assert elapsed >= 0.0
        assert isinstance(mem, float)
