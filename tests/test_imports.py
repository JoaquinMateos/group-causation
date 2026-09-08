import pytest


class TestCriticalImports:
    """Verifies that the broken imports identified in the audit are fixed."""

    def test_ng_vecci_contemporaneous_import(self):
        with pytest.raises(ImportError) as exc:
            import group_causation.group_causal_discovery.direction_extraction.NG_VecCI_contemporaneous  # noqa: F401
        msg = str(exc.value)
        assert "group_causal_discovery" in msg

    def test_ng_vecci_ts_import(self):
        with pytest.raises(ImportError) as exc:
            import group_causation.group_causal_discovery.direction_extraction.NG_VecCI_ts  # noqa: F401
        msg = str(exc.value)
        assert "group_causal_discovery" in msg

    def test_benchmark_base_import(self):
        try:
            from group_causation.benchmark.benchmark_base import BenchmarkBase  # noqa: F401
        except ImportError as exc:
            pytest.fail(f"benchmark_base import failed: {exc}")

    def test_benchmark_group_extraction_import(self):
        try:
            import group_causation.benchmark.benchmark_group_extraction  # noqa: F401
        except ImportError as exc:
            pytest.fail(f"benchmark_group_extraction import failed: {exc}")

    def test_all_init_files_load(self):
        modules = [
            "group_causation",
            "group_causation.groups_extraction",
            "group_causation.micro_causal_discovery",
            "group_causation.independence_tests",
            "group_causation.dimensionality_reduction",
            "group_causation.dimensionality_reduction.iVAE",
            "group_causation.group_causal_discovery",
            "group_causation.group_causal_discovery.direction_extraction",
            "group_causation.benchmark",
        ]
        for mod_name in modules:
            __import__(mod_name)
