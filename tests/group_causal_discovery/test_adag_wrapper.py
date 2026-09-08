import numpy as np
import pytest


@pytest.fixture
def data() -> np.ndarray:
    return np.random.randn(200, 4).astype(np.float64)


class TestAdagWrapper:
    def test_import_adag(self):
        try:
            from group_causation.group_causal_discovery.adag_wrapper import ADAGWrapper
            assert ADAGWrapper is not None
        except ImportError:
            pytest.skip("adag not installed")

    def test_import_adag_cd(self):
        try:
            from group_causation.group_causal_discovery.adag_wrapper import (
                ADAGGroupCausalDiscovery,
            )
            assert ADAGGroupCausalDiscovery is not None
        except ImportError:
            pytest.skip("adag not installed")
