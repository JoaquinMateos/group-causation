"""Aggregation-consistency scoring for group causal discovery.

Quantifies whether the conditional (in)dependencies discovered on aggregate
variables ``Z`` are reproduced by the underlying vector variables ``X``:

* ``c_ind``: fraction of macro-level independence statements that also hold
  on ``X`` (a low value means aggregation created false independencies and
  downstream causal discovery is deleting true edges).
* ``c_dep``: fraction of macro-level dependencies (adjacencies) that also hold
  on ``X``.
* ``ac``: their average, the joint Aggregation Consistency score.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

# (x_var, x_lag, y_var, y_lag, conditioning_set)
IndependenceStatement = tuple[int, int, int, int, list[tuple[int, int]]]
StatementTester = Callable[[IndependenceStatement], tuple[float, float]]


class InsufficientDataError(RuntimeError):
    """Raised by a statement tester when there are too few samples to decide."""


@dataclass(frozen=True)
class AggregationScore:
    """Aggregation-consistency scores returned by the evaluator."""

    c_ind: float
    c_dep: float
    ac: float


def consistency_ratio(consistent_flags: Iterable[bool]) -> float:
    """Fraction of consistent statements, defaulting to 1.0 when there are none."""
    flags = list(consistent_flags)
    return sum(flags) / len(flags) if flags else 1.0


def adjacency_statements(parents: Mapping[int, Sequence[tuple[int, int]]]) -> list[IndependenceStatement]:
    """Turn a positive-lag parents dict into dependency statements.

    Each adjacency ``(parent, lag) -> target`` is tested conditioning on the
    remaining parents of the target, matching the vector-level counterpart of
    the macro dependency.
    """
    statements: list[IndependenceStatement] = []
    for target, target_parents in parents.items():
        for parent in target_parents:
            conditioning = [other for other in target_parents if other != parent]
            statements.append((parent[0], parent[1], target, 0, conditioning))
    return statements


class AggregationConsistencyEvaluator:
    """Scores macro-level statements against vector-level conditional independence tests.

    A statement is *consistent* when the conclusion drawn at the aggregate
    level is reproduced at the vector level. Statements that cannot be tested
    (too few samples) are not penalized and count as consistent.

    Args:
        test_statement: Callable mapping a statement to ``(statistic, p_value)``.
            It should raise :class:`InsufficientDataError` when the sample is
            too small to reach a verdict.
        alpha: Significance level used to declare dependence.
    """

    def __init__(self, test_statement: StatementTester, alpha: float = 0.05):
        self._test_statement = test_statement
        self._alpha = alpha

    def c_ind(self, independencies: Iterable[IndependenceStatement]) -> float:
        """Consistency of macro-level independencies."""
        return consistency_ratio(self._is_consistent(statement, expected_independent=True)
                                 for statement in independencies)

    def c_dep(self, dependencies: Iterable[IndependenceStatement]) -> float:
        """Consistency of macro-level dependencies."""
        return consistency_ratio(self._is_consistent(statement, expected_independent=False)
                                 for statement in dependencies)

    def evaluate(self, independencies: Iterable[IndependenceStatement],
                 dependencies: Iterable[IndependenceStatement]) -> AggregationScore:
        """Compute ``c_ind``, ``c_dep`` and their average ``ac``."""
        c_ind = self.c_ind(independencies)
        c_dep = self.c_dep(dependencies)
        return AggregationScore(c_ind=c_ind, c_dep=c_dep, ac=(c_ind + c_dep) / 2.0)

    def _is_consistent(self, statement: IndependenceStatement, expected_independent: bool) -> bool:
        try:
            _, p_value = self._test_statement(statement)
        except InsufficientDataError:
            return True

        return p_value > self._alpha if expected_independent else p_value <= self._alpha
