from src.attacks.rays import compute_eggs_steps_to_try


def test_compute_eggs_steps_to_try():
    assert compute_eggs_steps_to_try(100) == 14
    assert compute_eggs_steps_to_try(1) == 1
