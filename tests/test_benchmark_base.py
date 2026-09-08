import numpy as np
import pytest

from group_causation.benchmark.benchmark_base import BenchmarkBase


class _ConcreteBenchmark(BenchmarkBase):
    def __init__(self, info_file="info.log", debug_file="debug.log", **kwargs):
        super().__init__(info_file=info_file, debug_file=debug_file)

    def run(self):
        return {"result": 42}

    def generate_datasets(self):
        pass

    def load_datasets(self):
        pass

    def test_particular_algorithm_particular_dataset(self, algo, dataset):
        pass


class TestBenchmarkBase:
    def test_default_init(self):
        inst = _ConcreteBenchmark()
        assert inst.verbose == 0

    def test_abstract_run_raises(self):
        class IncompleteBenchmark(BenchmarkBase):
            def __init__(self):
                super().__init__(info_file="info.log", debug_file="debug.log")

        with pytest.raises(TypeError):
            IncompleteBenchmark()

    def test_concrete_run(self):
        inst = _ConcreteBenchmark()
        result = inst.run()
        assert result == {"result": 42}
