import math

import pytest

from cloth.solver import (
    ClothError,
    collide,
    integrate,
    pin,
    self_collision,
    simulate_batch,
    solve_constraints,
    step,
    stretch_limit,
)


def _vec_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert math.isclose(got, want, rel_tol=0.0, abs_tol=tol), (actual, expected)


def _square():
    return {
        "positions": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        "constraints": [
            [0, 1, 1.0],
            [1, 3, 1.0],
            [3, 2, 1.0],
            [2, 0, 1.0],
        ],
    }


def _params(**overrides):
    params = {
        "dt": 1.0 / 60.0,
        "gravity": [0.0, 0.0, 0.0],
        "damping": 0.0,
        "iterations": 20,
        "stiffness": 1.0,
        "colliders": [],
        "radius": 0.0,
        "pinned": {},
    }
    params.update(overrides)
    return params


def test_semi_implicit_euler_avoids_stiff_divergence():
    positions = [[0.0, 0.0, 0.0]]
    velocities = [[0.0, 0.0, 0.0]]

    new_positions, new_velocities = integrate(
        positions, velocities, 0.1, [0.0, -10.0, 0.0], 0.0
    )

    _vec_close(new_velocities[0], [0.0, -1.0, 0.0])
    _vec_close(new_positions[0], [0.0, -0.1, 0.0])
    _vec_close(positions[0], [0.0, 0.0, 0.0])
    _vec_close(velocities[0], [0.0, 0.0, 0.0])


def test_pbd_correction_is_weighted_by_stiffness():
    out = solve_constraints(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0, 1, 1.0]],
        iterations=1,
        stiffness=0.5,
    )

    length = abs(out[1][0] - out[0][0])
    assert math.isclose(length, 1.5, rel_tol=0.0, abs_tol=1e-6)


def test_constraint_order_does_not_stretch_hem():
    count = 5
    positions = [[1.4 * i, 0.0, 0.0] for i in range(count)]
    constraints = [[i, i + 1, 1.0] for i in range(count - 1)]

    forward = solve_constraints(positions, constraints, iterations=300, stiffness=1.0)
    backward = solve_constraints(
        positions, list(reversed(constraints)), iterations=300, stiffness=1.0
    )

    for got, want in zip(backward, forward):
        _vec_close(got, want)
    span = forward[-1][0] - forward[0][0]
    assert math.isclose(span, 4.0, rel_tol=0.0, abs_tol=1e-3)


def test_collision_corrects_position_not_only_velocity():
    sphere = [{"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": 1.0}]
    out = collide([[0.5, 0.0, 0.0]], sphere, radius=0.1)
    distance = math.sqrt(sum(component * component for component in out[0]))
    assert math.isclose(distance, 1.1, rel_tol=0.0, abs_tol=1e-6)

    plane = [{"type": "plane", "normal": [1.0, 0.0, 0.0], "offset": 0.0}]
    out = collide([[-0.5, 0.0, 0.0]], plane, radius=0.1)
    assert math.isclose(out[0][0], 0.1, rel_tol=0.0, abs_tol=1e-6)


def test_pinned_points_return_to_original_after_solve():
    positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    solved = solve_constraints(positions, [[0, 1, 1.0]], iterations=10, stiffness=1.0)

    out = pin(solved, {0: [0.0, 0.0, 0.0]})

    _vec_close(out[0], [0.0, 0.0, 0.0])


def test_stretch_limit_clamps_extreme_extension():
    positions = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]

    out = stretch_limit([[0, 1, 1.0]], positions, max_ratio=2.0)

    length = abs(out[1][0] - out[0][0])
    assert math.isclose(length, 2.0, rel_tol=0.0, abs_tol=1e-6)


def test_self_collision_enforces_thickness():
    positions = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]

    out = self_collision(positions, thickness=0.5)

    delta = [out[1][i] - out[0][i] for i in range(3)]
    distance = math.sqrt(sum(component * component for component in delta))
    assert math.isclose(distance, 0.5, rel_tol=0.0, abs_tol=1e-6)


def test_damping_uses_time_constant_not_frame_count():
    _, one_step = integrate(
        [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], 0.1, [0.0, 0.0, 0.0], 1.0
    )
    assert math.isclose(one_step[0][0], math.exp(-0.1), rel_tol=0.0, abs_tol=1e-6)

    _, half_step = integrate(
        [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], 0.05, [0.0, 0.0, 0.0], 1.0
    )
    _, two_half_steps = integrate(
        [[0.0, 0.0, 0.0]], half_step, 0.05, [0.0, 0.0, 0.0], 1.0
    )
    assert math.isclose(
        two_half_steps[0][0], math.exp(-0.1), rel_tol=0.0, abs_tol=1e-6
    )


def test_batch_isolates_mutable_cloth_state():
    cloths = [
        {
            "positions": [[0.0, 0.0, 0.0]],
            "velocities": [[0.0, 0.0, 0.0]],
            "constraints": [],
        },
        {
            "positions": [[5.0, 5.0, 5.0]],
            "velocities": [[0.0, 0.0, 0.0]],
            "constraints": [],
        },
    ]

    out = simulate_batch(
        1,
        cloths,
        _params(dt=0.1, gravity=[0.0, -10.0, 0.0], iterations=0),
    )

    _vec_close(out["cloths"][0]["positions"][0], [0.0, -0.1, 0.0])
    _vec_close(out["cloths"][1]["positions"][0], [5.0, 5.0, 5.0])
    _vec_close(cloths[0]["positions"][0], [0.0, 0.0, 0.0])
    _vec_close(cloths[1]["positions"][0], [5.0, 5.0, 5.0])


def test_batch_aborts_and_rolls_back_on_non_finite_state():
    cloths = [
        {
            "positions": [[1e308, 0.0, 0.0]],
            "velocities": [[1e308, 0.0, 0.0]],
            "constraints": [],
        },
        {
            "positions": [[1.0, 1.0, 1.0]],
            "velocities": [[0.0, 0.0, 0.0]],
            "constraints": [],
        },
    ]

    with pytest.raises(ClothError):
        simulate_batch(2, cloths, _params(dt=1.0, iterations=0))

    _vec_close(cloths[0]["positions"][0], [1e308, 0.0, 0.0])
    _vec_close(cloths[1]["positions"][0], [1.0, 1.0, 1.0])


def test_rest_cloth_holds_position_after_one_step():
    cloth = _square()
    velocities = [[0.0, 0.0, 0.0] for _ in cloth["positions"]]

    out = step(cloth["positions"], velocities, cloth["constraints"], _params())

    for got, want in zip(out["positions"], cloth["positions"]):
        _vec_close(got, want)
    for got, want in zip(out["velocities"], velocities):
        _vec_close(got, want)


def test_zero_timestep_raises_cloth_error():
    with pytest.raises(ClothError):
        step([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [], _params(dt=0.0))


def test_empty_constraints_return_original_positions():
    positions = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]

    out = solve_constraints(positions, [], iterations=10, stiffness=1.0)

    for got, want in zip(out, positions):
        _vec_close(got, want)


def test_pinned_id_collection_is_not_mutated():
    pinned = {0: [0.0, 0.0, 0.0], 2: [1.0, 1.0, 1.0]}
    before = {index: list(value) for index, value in pinned.items()}

    out = pin([[9.0, 9.0, 9.0], [1.0, 0.0, 0.0], [8.0, 8.0, 8.0]], pinned)

    assert pinned == before
    _vec_close(out[0], [0.0, 0.0, 0.0])


def test_batch_success_preserves_cloth_order():
    cloths = [
        {
            "positions": [[0.0, 0.0, 0.0]],
            "velocities": [[0.0, 0.0, 0.0]],
            "constraints": [],
        },
        {
            "positions": [[10.0, 0.0, 0.0]],
            "velocities": [[0.0, 0.0, 0.0]],
            "constraints": [],
        },
    ]

    out = simulate_batch(3, cloths, _params(dt=0.1, iterations=0))

    assert out["frames"] == 3
    _vec_close(out["cloths"][0]["positions"][0], [0.0, 0.0, 0.0])
    _vec_close(out["cloths"][1]["positions"][0], [10.0, 0.0, 0.0])
