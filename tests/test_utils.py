import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from group_causation.utils import (
    changing_N_variables,
    changing_N_groups,
    changing_N_vars_per_group,
    changing_preselection_alpha,
    static_parameters,
    get_TP,
    get_FP,
    get_FN,
    get_precision,
    get_recall,
    get_f1,
    get_false_positive_ratio,
    window_to_summary_graph,
    split_lagged_and_contemporaneous,
    get_cpdag_and_edge_set,
    get_dag_edge_set,
    get_global_window_metrics,
    get_next_filename,
)


class TestParameterGenerators:
    def test_static_parameters(self):
        opts = {"N_vars": 10}
        params = {"alg": {"max_lag": 3}}
        results = list(static_parameters(opts, params))
        assert len(results) == 1
        assert results[0] == (params, opts)

    def test_changing_N_variables_default(self):
        opts = {"max_lag": 3}
        params = {"alg": {}}
        results = list(changing_N_variables(opts, params))
        assert len(results) == 5
        for _, opt in results:
            assert "N_vars" in opt

    def test_changing_N_variables_custom(self):
        opts = {"max_lag": 3}
        params = {"alg": {}}
        custom_list = [2, 8, 16]
        results = list(changing_N_variables(opts, params, list_N_variables=custom_list))
        assert len(results) == 3

    def test_changing_N_groups(self):
        opts = {}
        params = {"alg": {}}
        results = list(changing_N_groups(opts, params, list_N_groups=[3, 5], relation_vars_per_group=4))
        assert len(results) == 2

    def test_changing_preselection_alpha_missing_key(self):
        opts = {}
        params = {"alg": {}}
        with pytest.raises(KeyError):
            list(changing_preselection_alpha(opts, params, list_preselection_alpha=[0.1]))

    def test_changing_preselection_alpha_ok(self):
        opts = {}
        params = {"pcmci-modified": {}}
        results = list(changing_preselection_alpha(opts, params, list_preselection_alpha=[0.1, 0.2]))
        assert len(results) == 2

    def test_changing_N_vars_per_group_requires_N_groups(self):
        opts = {"N_groups": 5, "max_lag": 2}
        params = {"alg": {}}
        results = list(changing_N_vars_per_group(opts, params, list_N_vars_per_group=[3, 6]))
        assert len(results) == 2


class TestMetrics:
    def test_get_TP(self):
        gt = {(1, 2), (3, 4)}
        pred = {(1, 2), (5, 6)}
        assert get_TP(gt, pred) == 1

    def test_get_FP(self):
        gt = {(1, 2)}
        pred = {(1, 2), (3, 4)}
        assert get_FP(gt, pred) == 1

    def test_get_FN(self):
        gt = {(1, 2), (3, 4)}
        pred = {(1, 2)}
        assert get_FN(gt, pred) == 1

    @pytest.mark.parametrize(
        "tp, fp, fn, expected_precision, expected_recall, expected_f1",
        [
            (10, 0, 0, 1.0, 1.0, 1.0),
            (10, 5, 0, 10 / 15, 1.0, 2 * (10 / 15) / (10 / 15 + 1)),
            (0, 0, 10, 0.0, 0.0, 0.0),
            (0, 0, 0, 0.0, 0.0, 0.0),
        ],
    )
    def test_precision_recall_f1(self, tp, fp, fn, expected_precision, expected_recall, expected_f1):
        gt = set(range(tp + fn))
        pred = set(range(tp)) | set(range(tp, tp + fp))
        assert get_precision(gt, pred) == pytest.approx(expected_precision)
        assert get_recall(gt, pred) == pytest.approx(expected_recall)
        assert get_f1(gt, pred) == pytest.approx(expected_f1)

    def test_false_positive_ratio(self):
        gt = {(0, 0), (1, 1)}
        pred = {(0, 0), (2, 2), (3, 3)}
        fps = get_FP(gt, pred)
        n_nodes = 4
        total_possible = (n_nodes * (n_nodes - 1)) / 2
        actual_negatives = total_possible - len(gt)
        assert get_false_positive_ratio(gt, pred, n_nodes) == pytest.approx(fps / actual_negatives)

    def test_get_global_window_metrics_all_perfect(self):
        edges = {(1, 2)}
        gt_cpdag = MagicMock(spec=["shd"])
        pred_cpdag = MagicMock(spec=["shd"])
        gt_cpdag.shd.return_value = 0
        result = get_global_window_metrics(edges, edges, edges, edges, gt_cpdag, pred_cpdag)
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["shd"] == 0


class TestGraphUtils:
    def test_window_to_summary_graph(self, tiny_window_graph):
        result = window_to_summary_graph(tiny_window_graph)
        for node, parents in result.items():
            for _, lag in parents:
                assert lag == 0

    def test_window_to_summary_graph_removes_duplicates(self):
        graph = {0: [(1, -1), (1, -2)]}
        result = window_to_summary_graph(graph)
        assert len(result[0]) == 1

    def test_split_lagged_and_contemporaneous(self, sample_parents_dict):
        lagged, contemp = split_lagged_and_contemporaneous(sample_parents_dict)
        for parents_list in lagged.values():
            for _, lag in parents_list:
                assert lag < 0
        for parents_list in contemp.values():
            for _, lag in parents_list:
                assert lag == 0

    def test_get_cpdag_and_edge_set_silently_swallows_errors(self):
        graph = {0: [(1, -1)]}
        edge_set, cpdag = get_cpdag_and_edge_set(graph)
        assert isinstance(edge_set, set)
        assert cpdag is not None

    def test_get_dag_edge_set(self):
        graph = {0: [(1, -1), (2, 0)]}
        edges = get_dag_edge_set(graph)
        assert ("directed", (1, -1), 0) in edges
        assert ("directed", (2, 0), 0) in edges


class TestLoggingUtils:
    def test_get_next_filename_returns_base_when_no_conflict(self, tmp_path):
        name = get_next_filename(str(tmp_path / "debug.log"))
        assert name == str(tmp_path / "debug.log")

    def test_get_next_filename_increments_on_conflict(self, tmp_path):
        base = tmp_path / "debug.log"
        base.write_text("")
        name = get_next_filename(str(base))
        assert name != str(tmp_path / "debug.log")
        assert "debug" in name
