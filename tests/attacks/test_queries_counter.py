import torch

from src.attacks.queries_counter import AttackPhase, QueriesCounter


class DummyAttackPhase(AttackPhase):
    test = "test"


def test_queries_counter():
    counter = QueriesCounter(10)
    phase = DummyAttackPhase.test
    success = torch.tensor([True, True, False, False, True])
    distance = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    updated_counter = counter.increase(phase, success, distance)

    assert updated_counter.queries[phase] == 5
    assert updated_counter.unsafe_queries[phase] == 2
    assert updated_counter.total_queries == 5
    assert updated_counter.total_unsafe_queries == 2
    for i, (p, s, d) in enumerate(updated_counter.distances):
        assert p == phase
        assert s == success[i]
        assert d == distance[i]
    assert not updated_counter.is_out_of_queries()


def test_queries_counter_out_of_queries():
    counter = QueriesCounter(5)
    phase = DummyAttackPhase.test
    success = torch.tensor([True, True, False, False, True])
    distance = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    updated_counter = counter.increase(phase, success, distance)

    assert updated_counter.is_out_of_queries()
