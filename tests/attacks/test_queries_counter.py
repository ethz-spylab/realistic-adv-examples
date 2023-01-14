from dataclasses import replace

import torch

from src.attacks.queries_counter import AttackPhase, CurrentDistanceInfo, QueriesCounter


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


def test_queries_counter_expand():
    counter = QueriesCounter(10)
    phase = DummyAttackPhase.test
    success = torch.tensor([True])
    distance = torch.tensor([0.5])
    equivalent_simulated_queries = 2

    updated_counter = counter.increase(phase, success, distance, equivalent_simulated_queries)

    expanded_distances = [CurrentDistanceInfo(phase, True, 0.5, 0.5, 1)] * equivalent_simulated_queries
    assert updated_counter.expand_simulated_distances().distances == expanded_distances

    success = torch.tensor([False])
    distance = torch.tensor([0.4])
    equivalent_simulated_queries = 1
    new_updated_counter = updated_counter.increase(phase, success, distance, equivalent_simulated_queries)
    new_expanded_distances = expanded_distances + [CurrentDistanceInfo(phase, False, 0.4, 0.5, 1)]
    assert new_updated_counter.expand_simulated_distances().distances == new_expanded_distances


def test_current_distance_info_expand():
    distance_info = CurrentDistanceInfo(DummyAttackPhase.test, True, 0.5, 0.5, 3)
    assert distance_info.expand_equivalent_queries() == [replace(distance_info, equivalent_simulated_queries=1)] * 3
