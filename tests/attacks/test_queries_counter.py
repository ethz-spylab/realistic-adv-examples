import numpy as np

from src.attacks.queries_counter import AttackPhase, QueriesCounter


class DummyAttackPhase(AttackPhase):
    test = "test"


def test_queries_counter():
    counter = QueriesCounter()
    phase = DummyAttackPhase.test
    success = np.array([True, True, False, False, True])

    updated_counter = counter.increase(phase, success)

    assert updated_counter.queries[phase] == 5
    assert updated_counter.unsafe_queries[phase] == 2
    assert updated_counter.total_queries == 5
    assert updated_counter.total_unsafe_queries == 2
