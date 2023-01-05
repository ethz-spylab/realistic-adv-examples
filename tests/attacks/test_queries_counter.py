import torch

from src.attacks.queries_counter import AttackPhase, QueriesCounter


class DummyAttackPhase(AttackPhase):
    test = "test"


def test_queries_counter():
    counter = QueriesCounter(10)
    phase = DummyAttackPhase.test
    success = torch.tensor([True, True, False, False, True])
    distance = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])

    updated_counter = counter.increase(phase, success, distance)

    assert updated_counter.queries[phase] == 5
    assert updated_counter.unsafe_queries[phase] == 2
    assert updated_counter.total_queries == 5
    assert updated_counter.total_unsafe_queries == 2
    assert updated_counter.best_distance == distance.min().item()
    for i, distance_info in enumerate(updated_counter.distances):
        assert distance_info.phase == phase
        assert distance_info.safe == success[i]
        assert distance_info.distance == distance[i]
        best_distance_so_far = min(
            map(lambda t: t[1], filter(lambda t: t[0], zip(success[:i + 1].tolist(), distance[:i + 1].tolist()))))
        assert distance_info.best_distance == best_distance_so_far
    assert not updated_counter.is_out_of_queries()


def test_queries_counter_out_of_queries():
    counter = QueriesCounter(5)
    phase = DummyAttackPhase.test
    success = torch.tensor([True, True, False, False, True])
    distance = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    updated_counter = counter.increase(phase, success, distance)

    assert updated_counter.is_out_of_queries()
