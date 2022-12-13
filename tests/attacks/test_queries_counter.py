import torch

from src.attacks.queries_counter import AttackPhase, QueriesCounter


class DummyAttackPhase(AttackPhase):
    test = "test"


def test_queries_counter():
    counter = QueriesCounter(10)
    phase = DummyAttackPhase.test
    success = torch.tensor([True, True, False, False, True])

    updated_counter = counter.increase(phase, success)

    assert updated_counter.queries[phase] == 5
    assert updated_counter.unsafe_queries[phase] == 2
    assert updated_counter.total_queries == 5
    assert updated_counter.total_unsafe_queries == 2
    assert not updated_counter.is_out_of_queries()
    

def test_queries_counter_out_of_queries():
    counter = QueriesCounter(5)
    phase = DummyAttackPhase.test
    success = torch.tensor([True, True, False, False, True])

    updated_counter = counter.increase(phase, success)

    assert updated_counter.is_out_of_queries()
